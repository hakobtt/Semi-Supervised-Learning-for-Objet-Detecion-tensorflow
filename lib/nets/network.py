# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from model.nms_wrapper import nms

import numpy as np

from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from layer_utils.proposal_layer import proposal_layer, proposal_layer_tf
from layer_utils.proposal_top_layer import proposal_top_layer, proposal_top_layer_tf
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes

from model.config import cfg

from model.bbox_transform import clip_boxes1, bbox_transform_inv


class Network(object):
    def __init__(self):
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._gt_image = None
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}
        self.store_new_vals = True

    def _add_gt_image(self):
        # add back mean
        image = self.__image + cfg.PIXEL_MEANS
        # BGR to RGB (opencv uses BGR)
        resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
        self._gt_image = tf.reverse(resized, axis=[-1])

    def _add_gt_image_summary(self):
        # use a customized visualization function to visualize the boxes
        if self._gt_image is None:
            self._add_gt_image()
        image = tf.py_func(draw_bounding_boxes,
                           [self._gt_image, self._gt_boxes, self._im_info],
                           tf.float32, name="gt_boxes")

        return tf.summary.image('GROUND_TRUTH', image)

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:
            # change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped = tf.reshape(to_caffe,
                                  tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois, rpn_scores = proposal_top_layer_tf(
                    rpn_cls_prob,
                    rpn_bbox_pred,
                    self._im_info,
                    self._feat_stride,
                    self._anchors,
                    self._num_anchors
                )
            else:
                rois, rpn_scores = tf.py_func(proposal_top_layer,
                                              [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                               self._feat_stride, self._anchors, self._num_anchors],
                                              [tf.float32, tf.float32], name="proposal_top")

            rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
            rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois, rpn_scores = proposal_layer_tf(
                    rpn_cls_prob,
                    rpn_bbox_pred,
                    self._im_info,
                    self._mode,
                    self._feat_stride,
                    self._anchors,
                    self._num_anchors
                )
            else:
                rois, rpn_scores = tf.py_func(proposal_layer,
                                              [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                               self._feat_stride, self._anchors, self._num_anchors],
                                              [tf.float32, tf.float32], name="proposal")

            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

    # Only use it if you have roi_pooling op written in tf.image
    def _roi_pool_layer(self, bootom, rois, name):
        with tf.variable_scope(name) as scope:
            return tf.image.roi_pooling(bootom, rois,
                                        pooled_height=cfg.POOLING_SIZE,
                                        pooled_width=cfg.POOLING_SIZE,
                                        spatial_scale=1. / 16.)[0]

    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bounding boxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.POOLING_SIZE * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                             name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')

    def _dropout_layer(self, bottom, name, ratio=0.5):
        return tf.nn.dropout(bottom, ratio, name=name)

    def _anchor_target_layer(self, rpn_cls_score, name):
        with tf.variable_scope(name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32],
                name="anchor_target")

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                name="proposal_target")

            rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
            roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
            labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
            bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
            self._score_summaries.update(self._proposal_targets)

            return rois, roi_scores

    def _anchor_component(self):
        with tf.variable_scope('ANCHOR_' + self._tag) as scope:
            # just to get the shape right
            height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
            width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
            if cfg.USE_E2E_TF:
                anchors, anchor_length = generate_anchors_pre_tf(
                    height,
                    width,
                    self._feat_stride,
                    self._anchor_scales,
                    self._anchor_ratios
                )
            else:
                anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                                    [height, width,
                                                     self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                                    [tf.float32, tf.int32], name="generate_anchors")
            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchor_length = anchor_length

    def _build_network(self, x, is_training=True):
        # select initializers
        if cfg.TRAIN.TRUNCATED:
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

        net_conv = self._image_to_head(x, is_training)
        with tf.variable_scope(self._scope, self._scope):
            # build the anchors for the image
            self._anchor_component()
            # region proposal network
            rois = self._region_proposal(net_conv, is_training, initializer)
            # region of interest pooling
            if cfg.POOLING_MODE == 'crop':
                pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
            else:
                raise NotImplementedError

        fc7 = self._head_to_tail(pool5, is_training)
        with tf.variable_scope(self._scope, self._scope):
            # region classification
            cls_prob, bbox_pred = self._region_classification(fc7, is_training,
                                                              initializer, initializer_bbox)

        self._score_summaries.update(self._predictions)

        return rois, cls_prob, bbox_pred

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box

    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('LOSS_' + self._tag) as scope:
            # RPN, class loss
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            # RCNN, class loss
            cls_score = self._predictions["cls_score"]
            label = tf.reshape(self._proposal_targets["labels"], [-1])
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

            # RCNN, bbox loss
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
            if self.store_new_vals:
                self._losses['cross_entropy'] = cross_entropy
                self._losses['loss_box'] = loss_box
                self._losses['rpn_cross_entropy'] = rpn_cross_entropy
                self._losses['rpn_loss_box'] = rpn_loss_box

                loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
                regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
                self._losses['regularization_loss'] = regularization_loss
                self._losses['total_loss'] = loss + regularization_loss

            self._event_summaries.update(self._losses)

        return loss

    def _region_proposal(self, net_conv, is_training, initializer):
        rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
                          scope="rpn_conv/3x3")
        self._act_summaries.append(rpn)
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_cls_score')
        # change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
        if is_training:
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
            # Try to have a deterministic order for the computing graph, for reproducibility
            with tf.control_dependencies([rpn_labels]):
                rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
        else:
            if cfg.TEST.MODE == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.TEST.MODE == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError
        if self.store_new_vals:
            self._predictions["rpn_cls_score"] = rpn_cls_score
            self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
            self._predictions["rpn_cls_prob"] = rpn_cls_prob
            self._predictions["rpn_cls_pred"] = rpn_cls_pred
            self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
            self._predictions["rois"] = rois

        return rois

    def _region_classification(self, fc7, is_training, initializer, initializer_bbox):
        cls_score = slim.fully_connected(fc7, self._num_classes,
                                         weights_initializer=initializer,
                                         trainable=is_training,
                                         activation_fn=None, scope='cls_score')
        cls_prob = self._softmax_layer(cls_score, "cls_prob")
        cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
        bbox_pred = slim.fully_connected(fc7, self._num_classes * 4,
                                         weights_initializer=initializer_bbox,
                                         trainable=is_training,
                                         activation_fn=None, scope='bbox_pred')
        if self.store_new_vals:
            self._predictions["cls_score"] = cls_score
            self._predictions["cls_pred"] = cls_pred
            self._predictions["cls_prob"] = cls_prob
            self._predictions["bbox_pred"] = bbox_pred

        return cls_prob, bbox_pred

    def _image_to_head(self, x, is_training, reuse=None):
        raise NotImplementedError

    def _head_to_tail(self, pool5, is_training, reuse=None):
        raise NotImplementedError

    def clear_all(self):
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._gt_image = None
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}
        self.store_new_vals = True

    def get_logits(self, x):
        with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=self.weights_regularizer,
                       biases_regularizer=self.biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0), reuse=tf.AUTO_REUSE):
            rois, cls_prob, bbox_pred = self._build_network(x, self.training)
            return cls_prob

    def create_perturbation_func(self):
        self.create_training_testing_vars()
        old_clas_prob = self._predictions["cls_prob"]
        self.clear_all()
        self.store_new_vals = True
        self.pert_image = self.__image
        self.pert_im_info = self._im_info
        self.pert_gt_boxes = self._gt_boxes
        self._perturbation = generate_virtual_adversarial_perturbation(self.__image, old_clas_prob,
                                                                       self.get_logits)
        stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
        means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
        self._predictions["vat_bbox_pred"] = self._predictions["bbox_pred"] + means
        self._predictions["vat_bbox_pred"] *= stds
        self.pert_predictions = {}
        for k, v in self._predictions.items():
            self.pert_predictions[k] = v

    def create_training_testing_vars(self):
        self.training = self._mode == 'TRAIN'
        self.testing = self._mode == 'TEST'
        self.store_new_vals = True
        self.get_logits(self.__image)
        # list as many types of layers as possible, even if they are not used now
        layers_to_output = {'rois': self._predictions["rois"]}
        for var in tf.trainable_variables():
            self._train_summaries.append(var)
        if self.testing:
            stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
            means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
            self._predictions["bbox_pred"] += means
            self._predictions["bbox_pred"] *= stds
        else:
            self._add_losses()
            layers_to_output.update(self._losses)

        return layers_to_output

    def create_summaries(self):
        val_summaries = []
        with tf.device("/cpu:0"):
            val_summaries.append(self._add_gt_image_summary())
            for key, var in self._event_summaries.items():
                val_summaries.append(tf.summary.scalar('Labeled ' + key, var))
            for key, var in self._score_summaries.items():
                self._add_score_summary(key, var)
            for var in self._act_summaries:
                self._add_act_summary(var)
            for var in self._train_summaries:
                self._add_train_summary(var)
        self._summary_op = tf.summary.merge_all()
        self._summary_op_val = tf.summary.merge(val_summaries)

    def create_network_for_inps(self, image, im_info, gt_boxes):
        self.__image = image
        self._im_info = im_info
        self._gt_boxes = gt_boxes
        return self.create_training_testing_vars()

    def create_semi_supervised_loss(self):
        self.clear_all()
        ALPHA = 1.0
        self.labeled_image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self.labeled_im_info = tf.placeholder(tf.float32, shape=[3])
        self.labeled_gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.alpha = tf.placeholder(tf.float32, shape=[1])

        self.create_network_for_inps(self.labeled_image, self.labeled_im_info, self.labeled_gt_boxes)
        labeled_loses = {}
        for k, v in self._losses.items():
            labeled_loses[k] = v
        self.semi_supervised_losses = {}
        val_summaries = []
        with tf.device("/cpu:0"):
            val_summaries.append(self._add_gt_image_summary())
            for key, var in self._event_summaries.items():
                val_summaries.append(tf.summary.scalar('Labeled ' + key, var))
        self._summary_op_val = tf.summary.merge(val_summaries)
        self.clear_all()
        self.orig_image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self.orig_im_info = tf.placeholder(tf.float32, shape=[3])
        self.orig_gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.create_network_for_inps(self.orig_image, self.orig_im_info, self.orig_gt_boxes)
        orig_losses = {}
        for k, v in self._losses.items():
            orig_losses[k] = self.alpha[0] * v
        for k in labeled_loses.keys():
            self.semi_supervised_losses[k] = labeled_loses[k] + orig_losses[k]

        with tf.device("/cpu:0"):
            val_summaries.append(self._add_gt_image_summary())
            for key, var in self._event_summaries.items():
                val_summaries.append(tf.summary.scalar('Orig ' + key, var))
            for key, var in self.semi_supervised_losses.items():
                val_summaries.append(tf.summary.scalar('SumLoss ' + key, var))
        self._summary_op = tf.summary.merge(val_summaries)

        return self.semi_supervised_losses

    def create_architecture(self, mode, num_classes, tag=None,
                            anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):

        self.__image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self._tag = tag

        self._num_classes = num_classes
        self._mode = mode
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)

        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)

        self._num_anchors = self._num_scales * self._num_ratios

        self.training = mode == 'TRAIN'
        self.testing = mode == 'TEST'

        assert tag != None

        # handle most of the regularizers here
        self.weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            self.biases_regularizer = self.weights_regularizer
        else:
            self.biases_regularizer = tf.no_regularizer

    def get_variables_to_restore(self, variables, var_keep_dic):
        raise NotImplementedError

    def fix_variables(self, sess, pretrained_model):
        raise NotImplementedError

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    # only useful during testing mode
    def extract_head(self, sess, image):
        feed_dict = {self.__image: image}
        feat = sess.run(self._layers["head"], feed_dict=feed_dict)
        return feat

    # only useful during testing mode
    def test_image(self, sess, image, im_info):
        feed_dict = {self.__image: image,
                     self._im_info: im_info,
                     }

        cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                         self._predictions['cls_prob'],
                                                         self._predictions['bbox_pred'],
                                                         self._predictions['rois']],
                                                        feed_dict=feed_dict)
        return cls_score, cls_prob, bbox_pred, rois

    def test_image_on_batch(self, sess, images, im_infos):
        feed_dict = {self.batch_image: images,
                     self.batch_im_info: im_infos}

        cls_score, cls_prob, bbox_pred, rois = sess.run([self.batch_predictions["cls_score"],
                                                         self.batch_predictions['cls_prob'],
                                                         self.batch_predictions['bbox_pred'],
                                                         self.batch_predictions['rois']],
                                                        feed_dict=feed_dict)
        return cls_score, cls_prob, bbox_pred, rois

    def get_summary(self, sess, blobs):
        feed_dict = {self.__image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        # feed_dict = {self.__image: blobs['data'], self._im_info: blobs['im_info'],
        #             self._gt_boxes: blobs['gt_boxes']}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def get_summary_semi_supervised(self, sess, blobs):
        feed_dict = {self.labeled_image: blobs['data'], self.labeled_im_info: blobs['im_info'],
                     self.labeled_gt_boxes: blobs['gt_boxes']}
        # feed_dict = {self.__image: blobs['data'], self._im_info: blobs['im_info'],
        #             self._gt_boxes: blobs['gt_boxes']}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def get_perturbation(self, sess, blobs):
        feed_dict = {self.pert_image: blobs['data'], self.pert_im_info: blobs['im_info'],
                     self.pert_gt_boxes: np.zeros(blobs['gt_boxes'].shape)}
        perturbation, cls_prob, bbox_pred, rois = sess.run([self._perturbation,
                                                            self.pert_predictions['cls_prob'],
                                                            self.pert_predictions['vat_bbox_pred'],
                                                            self.pert_predictions['rois']],
                                                           feed_dict=feed_dict)
        boxes = rois[:, 1:5]
        scores = np.reshape(cls_prob, [cls_prob.shape[0], -1])
        bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes1(pred_boxes, blobs['data'][0].shape)
        CONF_THRESH = 0.8
        NMS_THRESH = 0.25
        results_gt_boxes = []
        for cls_ind in range(self._num_classes - 1):
            cls_ind += 1  # because we skipped background
            cls_boxes = pred_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            for ind in inds:
                gt_box = dets[ind]
                gt_box[-1] = cls_ind
                results_gt_boxes.append(gt_box)

        return perturbation, results_gt_boxes

    def train_step(self, sess, blobs, train_op):
        feed_dict = {self.__image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                            self._losses['rpn_loss_box'],
                                                                            self._losses['cross_entropy'],
                                                                            self._losses['loss_box'],
                                                                            self._losses['total_loss'],
                                                                            train_op],
                                                                           feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

    def train_step_semi_supervised(self, sess, blobs_labeled, blobs_orig, alpha, train_op):
        feed_dict = {self.labeled_image: blobs_labeled['data'], self.labeled_im_info: blobs_labeled['im_info'],
                     self.labeled_gt_boxes: blobs_labeled['gt_boxes'],
                     self.orig_image: blobs_orig['data'], self.orig_im_info: blobs_orig['im_info'],
                     self.orig_gt_boxes: blobs_orig['gt_boxes'],
                     self.alpha: [alpha]
                     }
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run(
            [self.semi_supervised_losses["rpn_cross_entropy"],
             self.semi_supervised_losses['rpn_loss_box'],
             self.semi_supervised_losses['cross_entropy'],
             self.semi_supervised_losses['loss_box'],
             self.semi_supervised_losses['total_loss'],
             train_op],
            feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

    def train_step_with_summary_semi_supervised(self, sess, blobs_labeled, blobs_orig, alpha, train_op):
        feed_dict = {self.labeled_image: blobs_labeled['data'], self.labeled_im_info: blobs_labeled['im_info'],
                     self.labeled_gt_boxes: blobs_labeled['gt_boxes'],
                     self.orig_image: blobs_orig['data'], self.orig_im_info: blobs_orig['im_info'],
                     self.orig_gt_boxes: blobs_orig['gt_boxes'],
                     self.alpha: [alpha]
                     }
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run(
            [self.semi_supervised_losses["rpn_cross_entropy"],
             self.semi_supervised_losses['rpn_loss_box'],
             self.semi_supervised_losses['cross_entropy'],
             self.semi_supervised_losses['loss_box'],
             self.semi_supervised_losses['total_loss'],
             self._summary_op,
             train_op],
            feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = {self.__image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                     self._losses['rpn_loss_box'],
                                                                                     self._losses['cross_entropy'],
                                                                                     self._losses['loss_box'],
                                                                                     self._losses['total_loss'],
                                                                                     self._summary_op,
                                                                                     train_op],
                                                                                    feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

    def train_step_no_return(self, sess, blobs, train_op):
        feed_dict = {self.__image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        sess.run([train_op], feed_dict=feed_dict)


def generate_virtual_adversarial_perturbation(x, logit, forward):
    """Generate an adversarial perturbation.

    Args:
        x: Model inputs.
        logit: Original model output without perturbation.
        forward: Callable which computs logits given input.
        hps: Model hyperparameters.

    Returns:
        Aversarial perturbation to be applied to x.
    """
    d = tf.random.normal(shape=tf.shape(x))
    vat_epsilon = 200
    vat_xi = 0.5
    for _ in range(1):
        d = vat_xi * get_normalized_vector(d)
        logit_p = logit
        logit_m = forward(x + d)
        dist = kl_divergence_with_logit(logit_p, logit_m)
        grad = tf.gradients(tf.reduce_mean(dist), [d], aggregation_method=2)[0]
        d = tf.stop_gradient(grad)

    return vat_epsilon * get_normalized_vector(d)


def kl_divergence_with_logit(q_logit, p_logit):
    """Compute the per-element KL-divergence of a batch."""
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_sum(q * logsoftmax(q_logit), 1)
    qlogp = tf.reduce_sum(q * logsoftmax(p_logit), 1)
    return qlogq - qlogp


def get_normalized_vector(d):
    """Normalize d by infinity and L2 norms."""
    d /= 1e-12 + tf.reduce_max(
        tf.abs(d), list(range(1, len(d.get_shape()))), keepdims=True
    )
    d /= tf.sqrt(
        1e-6
        + tf.reduce_sum(
            tf.pow(d, 2.0), list(range(1, len(d.get_shape()))), keepdims=True
        )
    )
    return d


def logsoftmax(x):
    """Compute log-domain softmax of logits."""
    xdev = x - tf.reduce_max(x, 1, keepdims=True)
    lsm = xdev - tf.math.log(tf.reduce_sum(tf.exp(xdev), 1, keepdims=True))
    return lsm

# def create_two_batch_inp(self):
#    self.batch_image = tf.placeholder(tf.float32, shape=[2, None, None, 3])
#    self.batch_im_info = tf.placeholder(tf.float32, shape=[2, 3])
#    # tf.py_function(self.append_new_im, [], Tout=tf.float32)
#    self.append_new_im()
#
#
# def append_new_im(self):
#    cls_scores = []
#    cls_probs = []
#    bbox_preds = []
#    rois = []
#    self.batch_predictions = {}
#    for i in range(self.batch_image.shape[0]):
#        self.__image = tf.expand_dims(self.batch_image[i], 0)
#        self._im_info = self.batch_im_info[i]
#        self.create_training_testing_vars()
#        cls_scores.append(tf.expand_dims(self._predictions["cls_score"], 0))
#        cls_probs.append(tf.expand_dims(self._predictions['cls_prob'], 0))
#        bbox_preds.append(tf.expand_dims(self._predictions['bbox_pred'], 0))
#        rois.append(tf.expand_dims(self._predictions['rois'], 0))
#    self.batch_predictions["cls_score"] = tf.concat(cls_scores, 0)
#    self.batch_predictions['cls_prob'] = tf.concat(cls_probs, 0)
#    self.batch_predictions['bbox_pred'] = tf.concat(bbox_preds, 0)
#    self.batch_predictions['rois'] = tf.concat(rois, 0)
#
