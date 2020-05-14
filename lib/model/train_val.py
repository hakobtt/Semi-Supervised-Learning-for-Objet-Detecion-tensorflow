# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Zheqi He
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer

try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import os
import sys
import glob
import time

import cv2

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


class SolverWrapper(object):
    """
      A wrapper class for the training process
    """

    def __init__(self, sess, network, imdb, roidb, valroidb, output_dir, tbdir, pretrained_model=None):
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.valroidb = valroidb
        self.output_dir = output_dir
        self.tbdir = tbdir
        # Simply put '_val' at the end to save the summaries from the validation set
        self.tbvaldir = tbdir + '_val'
        if not os.path.exists(self.tbvaldir):
            os.makedirs(self.tbvaldir)
        self.pretrained_model = pretrained_model

    def snapshot(self, sess, iter):
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # Also store some meta information, random state, etc.
        nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
        nfilename = os.path.join(self.output_dir, nfilename)
        # current state of numpy random
        st0 = np.random.get_state()
        # Dump the meta info
        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.data_layer_overall, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.data_layer_labeled, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.data_layer_val, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename

    def from_snapshot(self, sess, sfile, nfile):
        print('Restoring model snapshots from {:s}'.format(sfile))
        self.saver.restore(sess, sfile)
        print('Restored.')
        # Needs to restore the other hyper-parameters/states for training, (TODO xinlei) I have
        # tried my best to find the random states so that it can be recovered exactly
        # However the Tensorflow state is currently not available
        with open(nfile, 'rb') as fid:
            st0 = pickle.load(fid)
            # cur = pickle.load(fid)
            # perm = pickle.load(fid)
            data_layer_overal = pickle.load(fid)
            data_layer_labeled = pickle.load(fid)
            data_layer_val = pickle.load(fid)
            last_snapshot_iter = pickle.load(fid)

            np.random.set_state(st0)
            self.data_layer_overall = data_layer_overal
            self.data_layer_labeled = data_layer_labeled
            self.data_layer_val = data_layer_val
        return last_snapshot_iter

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

    def construct_graph(self, sess):
        with sess.graph.as_default():
            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)
            # Build the main computation graph
            self.net.create_architecture('TRAIN', self.imdb.num_classes, tag='default',
                                         anchor_scales=cfg.ANCHOR_SCALES,
                                         anchor_ratios=cfg.ANCHOR_RATIOS)

            if self.semi_supervised:
                self.net.create_perturbation_func()
                layers = self.net.create_semi_supervised_loss()
            else:
                layers = self.net.create_training_testing_vars()
                self.net.create_summaries()

            # layers = self.net.create_training_testing_vars()
            # Define the loss
            loss = layers['total_loss']
            # Set learning rate and momentum
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
            self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

            # Compute the gradients with regard to the loss
            gvs = self.optimizer.compute_gradients(loss)
            # Double the gradient of the bias if set
            if cfg.TRAIN.DOUBLE_BIAS:
                final_gvs = []
                with tf.variable_scope('Gradient_Mult') as scope:
                    for grad, var in gvs:
                        scale = 1.
                        if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
                            scale *= 2.
                        if not np.allclose(scale, 1.0):
                            grad = tf.multiply(grad, scale)
                        final_gvs.append((grad, var))
                train_op = self.optimizer.apply_gradients(final_gvs)
            else:
                train_op = self.optimizer.apply_gradients(gvs)

            # We will handle the snapshots ourselves
            self.saver = tf.train.Saver(max_to_keep=100000)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph, )
            self.valwriter = tf.summary.FileWriter(self.tbvaldir)

        return lr, train_op

    def find_previous(self):
        sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)
        # Get the snapshot name in TensorFlow
        redfiles = []
        for stepsize in cfg.TRAIN.STEPSIZE:
            redfiles.append(os.path.join(self.output_dir,
                                         cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}.ckpt.meta'.format(stepsize + 1)))
        sfiles = [ss.replace('.meta', '') for ss in sfiles if ss not in redfiles]

        nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
        nfiles = glob.glob(nfiles)
        nfiles.sort(key=os.path.getmtime)
        redfiles = [redfile.replace('.ckpt.meta', '.pkl') for redfile in redfiles]
        nfiles = [nn for nn in nfiles if nn not in redfiles]

        lsf = len(sfiles)
        assert len(nfiles) == lsf

        return lsf, nfiles, sfiles

    def initialize(self, sess):
        # Initial file lists are empty
        np_paths = []
        ss_paths = []
        # Fresh train directly from ImageNet weights
        print('Loading initial model weights from {:s}'.format(self.pretrained_model))
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name='init'))
        var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
        # Get the variables to restore, ignoring the variables to fix
        variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)

        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, self.pretrained_model)
        print('Loaded.')
        # Need to fix the variables before loading, so that the RGB weights are changed to BGR
        # For VGG16 it also changes the convolutional weights fc6 and fc7 to
        # fully connected weights
        self.net.fix_variables(sess, self.pretrained_model)
        print('Fixed.')
        last_snapshot_iter = 0
        rate = cfg.TRAIN.LEARNING_RATE
        stepsizes = list(cfg.TRAIN.STEPSIZE)

        return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

    def restore(self, sess, sfile, nfile):
        # Get the most recent snapshot and restore
        np_paths = [nfile]
        ss_paths = [sfile]
        # Restore model from snapshots
        last_snapshot_iter = self.from_snapshot(sess, sfile, nfile)
        # Set the learning rate
        rate = cfg.TRAIN.LEARNING_RATE
        stepsizes = []
        for stepsize in cfg.TRAIN.STEPSIZE:
            if last_snapshot_iter > stepsize:
                rate *= cfg.TRAIN.GAMMA
            else:
                stepsizes.append(stepsize)

        return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

    def remove_snapshot(self, np_paths, ss_paths):
        to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            nfile = np_paths[0]
            os.remove(str(nfile))
            np_paths.remove(nfile)

        to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            sfile = ss_paths[0]
            # To make the code compatible to earlier versions of Tensorflow,
            # where the naming tradition for checkpoints are different
            if os.path.exists(str(sfile)):
                os.remove(str(sfile))
            else:
                os.remove(str(sfile + '.data-00000-of-00001'))
                os.remove(str(sfile + '.index'))
            sfile_meta = sfile + '.meta'
            os.remove(str(sfile_meta))
            ss_paths.remove(sfile)

    def separate_labeled_rois(self, use_all):
        if use_all:
            return self.roidb
        labeled_images_db = []
        _perm = np.random.permutation(np.arange(len(self.roidb)))
        inds_to_remove_from_image_db = []
        for class_id in range(1, self.imdb.num_classes):
            found_img_count = 0
            for i in _perm:
                roi = self.roidb[i]
                if class_id in roi['gt_classes'] and i not in inds_to_remove_from_image_db:
                    found_img_count += 1
                    labeled_images_db.append(roi)
                    inds_to_remove_from_image_db.append(i)
                if found_img_count >= cfg.TRAIN.LABELED_IMAGES_PER_CLASS:
                    break
        return labeled_images_db

    def train_model(self, sess, max_iters, semi_supervised, use_all=False):
        self.semi_supervised = semi_supervised
        # Build data layers for both training and validation set
        self.labeled_images_db = self.separate_labeled_rois(use_all)
        self.data_layer_overall = RoIDataLayer(self.roidb, self.imdb.num_classes)
        self.data_layer_labeled = RoIDataLayer(self.labeled_images_db, self.imdb.num_classes)
        self.data_layer_val = RoIDataLayer(self.valroidb, self.imdb.num_classes, random=True)
        self.visualize_perturbation = False
        use_image = True
        # Construct the computation graph
        lr, train_op = self.construct_graph(sess)

        # Find previous snapshots if there is any to restore from
        lsf, nfiles, sfiles = self.find_previous()

        # Initialize the variables or restore them from the last snapshot
        if lsf == 0:
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize(sess)
        else:
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(sess,
                                                                                   str(sfiles[-1]),
                                                                                   str(nfiles[-1]))
        timer = Timer()
        iter = last_snapshot_iter + 1
        last_summary_time = time.time()
        # Make sure the lists are not empty
        stepsizes.append(max_iters)
        stepsizes.reverse()
        next_stepsize = stepsizes.pop()
        while iter < max_iters + 1:
            # Learning rate

            if iter == next_stepsize + 1:
                # Add snapshot here before reducing the learning rate
                self.snapshot(sess, iter)
                rate *= cfg.TRAIN.GAMMA
                sess.run(tf.assign(lr, rate))
                next_stepsize = stepsizes.pop()

            timer.tic()
            # Get training data, one batch at a time
            cur_alpha = cfg.TRAIN.ALPHA
            labeled_blobs = self.data_layer_labeled.forward()
            if self.semi_supervised:
                orig_blobs = self.data_layer_overall.forward()
                perturbation, new_gt_boxes = self.net.get_perturbation(sess, orig_blobs)
                if self.visualize_perturbation:
                    img = (orig_blobs['data'][0] + cfg.PIXEL_MEANS).astype(np.uint8)
                    for rect in orig_blobs['gt_boxes']:
                        cv2.rectangle(img, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), [255, 0, 0])
                        cv2.putText(img, str(rect[-1]), (int(rect[0]), int(rect[1])), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('annotated_boxes', img)
                    cv2.waitKey(1)

                    perturbation_img = (255 * (perturbation[0] - perturbation[0].min()) / (
                        (perturbation[0].max() - perturbation[0].min()))).astype(np.uint8)
                    perturbated_img = (orig_blobs['data'][0] + perturbation[0] + cfg.PIXEL_MEANS).astype(np.uint8)
                    cv2.imshow('perturbated', perturbated_img)
                    cv2.imshow('perturbation', perturbation_img)
                    cv2.waitKey(1)
                    img = (orig_blobs['data'][0] + cfg.PIXEL_MEANS).astype(np.uint8)
                    for rect in new_gt_boxes:
                        cv2.rectangle(img, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), [255, 0, 0])
                        cv2.putText(img, str(rect[-1]), (int(rect[0]), int(rect[1])), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('new_detected_boxes', img)
                    cv2.waitKey(0)
                use_image = len(new_gt_boxes) > 0
                orig_blobs['data'] = orig_blobs['data'] + perturbation
                orig_blobs['gt_boxes'] = np.array(new_gt_boxes)

            if not use_image:
                orig_blobs = labeled_blobs
                cur_alpha = 0

            now = time.time()
            if now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
                # Compute the graph with summary
                if semi_supervised:
                    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = \
                        self.net.train_step_with_summary_semi_supervised(sess, labeled_blobs, orig_blobs, cur_alpha,
                                                                         train_op)
                else:
                    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = \
                        self.net.train_step_with_summary(sess, labeled_blobs, train_op)
                self.writer.add_summary(summary, float(iter))
                # Also check the summary on the validation set
                blobs_val = self.data_layer_val.forward()
                if semi_supervised:
                    summary_val = self.net.get_summary_semi_supervised(sess, blobs_val)
                else:
                    summary_val = self.net.get_summary(sess, blobs_val)
                self.valwriter.add_summary(summary_val, float(iter), )
                last_summary_time = now
            else:
                # Compute the graph without summary
                if semi_supervised:
                    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = \
                        self.net.train_step_semi_supervised(sess, labeled_blobs, orig_blobs, cur_alpha, train_op)
                else:
                    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = \
                        self.net.train_step(sess, labeled_blobs, train_op)
            timer.toc()

            # Display training information
            if iter == 1 or iter % (cfg.TRAIN.DISPLAY) == 0:
                print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                      '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> lr: %f' % \
                      (iter, max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, lr.eval()))
                print('speed: {:.3f}s / iter'.format(timer.average_time))

            # Snapshotting
            if iter == 1 or iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                ss_path, np_path = self.snapshot(sess, iter)
                np_paths.append(np_path)
                ss_paths.append(ss_path)

                # Remove the old snapshots if there are too many
                if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
                    self.remove_snapshot(np_paths, ss_paths)

            iter += 1

        if last_snapshot_iter != iter - 1:
            self.snapshot(sess, iter - 1)

        self.writer.close()
        self.valwriter.close()


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb


def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after))
    return filtered_roidb


def train_net(network, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=None,
              max_iters=40000):
    """Train a Faster R-CNN network."""
    roidb = filter_roidb(roidb)
    valroidb = filter_roidb(valroidb)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapper(sess, network, imdb, roidb, valroidb, output_dir, tb_dir,
                           pretrained_model=pretrained_model)
        print('Solving...')
        sw.train_model(sess, 35000, True)
        print('done solving')
