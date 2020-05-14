# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1
import numpy as np
from sklearn import utils as sk_utils


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--model', dest='model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152, mobile',
                        default='res50', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # args.model = "/home/hakob/Desktop/tf-faster-rcnn/output/res101/voc_2007_trainval/only_labeled_3_per_class_alpha_1.0/res101_faster_rcnn_iter_20000.ckpt"
    # args.model = "/home/hakob/Desktop/tf-faster-rcnn/output/res101/voc_2007_trainval/from_imagenet_3_per_class_alpha_1.0/res101_faster_rcnn_iter_10000.ckpt"
    # args.model = "/home/hakob/Desktop/tf-faster-rcnn/output/res101/voc_2007_trainval/from_10000_trained_3_per_class_alpha_1.0/res101_faster_rcnn_iter_22000.ckpt"
    # args.model = "/home/hakob/Desktop/tf-faster-rcnn/output/res101/voc_2007_trainval/from_10000_trained10_per_class_alpha_1.0/res101_faster_rcnn_iter_17000.ckpt"
    # args.model = "/home/hakob/Desktop/tf-faster-rcnn/output/res101/voc_2007_trainval/on_labeled_10_from_image_net_semi_sup/res101_faster_rcnn_iter_15000.ckpt"
    # args.model = "/home/hakob/Desktop/tf-faster-rcnn/output/res101/voc_2007_trainval/default111/res101_faster_rcnn_iter_60000.ckpt"
    # args.model = "/home/hakob/Desktop/tf-faster-rcnn/output/res101/voc_2007_trainval/on_labeled_semi_supervised_good/res101_faster_rcnn_iter_79000.ckpt"
    args.model = "/home/hakob/Desktop/tf-faster-rcnn/output/res101/voc_2007_trainval/fully_sup/res101_faster_rcnn_iter_60000.ckpt"
    # args.model = "/home/hakob/Desktop/tf-faster-rcnn/pretrained/tf-faster-rcnn/res101/res101_faster_rcnn_iter_110000.ckpt"

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    # if has model, get the name from it
    # if does not, then just use the initialization weights
    if args.model:
        filename = os.path.splitext(os.path.basename(args.model))[0]
    else:
        filename = os.path.splitext(os.path.basename(args.weight))[0]

    tag = args.tag
    tag = tag if tag else 'default'
    wait_name = filename
    filename = tag + '/' + filename

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    imagesetfile = imdb.get_image_set_file_name()
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    output_dir = get_output_dir(imdb, filename)
    num_tests = 80
    good_set = 0.0
    per_cls_res = {}
    every_it_map_1n = []
    every_it_map_2n = []
    for i in range(num_tests):
        cur_keys = sk_utils.resample(imagenames, n_samples=len(imagenames))
        cur_keys_s = set(cur_keys)
        cur_keys = list(cur_keys_s)
        # cur_keys = imagenames
        res1, clss = imdb._do_python_eval(output_dir, annot_file_tag="res101_faster_rcnn_iter_60000",
                                          image_set=cur_keys)
        every_it_map_1n.append(res1)

    n1_5per = np.percentile(every_it_map_1n, 5, axis=0)
    n1_95per = np.percentile(every_it_map_1n, 95, axis=0)
    print("semi_sup")
    for i in range(len(clss)):
        print(('{} {:.1f} - {:.2f}'.format(clss[i], 100*n1_5per[i], 100*n1_95per[i])))
    print("fn ", np.percentile(np.mean(every_it_map_1n, axis=1), 5),
          np.percentile(np.mean(every_it_map_1n, axis=1), 95))
    exit()
    for i in range(num_tests):
        cur_keys = sk_utils.resample(imagenames, n_samples=len(imagenames))
        cur_keys_s = set(cur_keys)
        cur_keys = list(cur_keys_s)
        # cur_keys = imagenames
        res1, clss = imdb._do_python_eval(output_dir, annot_file_tag="res101_faster_rcnn_iter_74000",
                                          image_set=cur_keys)
        res2, clss = imdb._do_python_eval(output_dir, annot_file_tag="res101_faster_rcnn_iter_70000",
                                          image_set=cur_keys)
        for r1, r2, c in zip(res1, res2, clss):
            if c not in per_cls_res:
                per_cls_res[c] = 0
            if r1 > r2:
                per_cls_res[c] += 1
        isgood = np.mean(res1) > np.mean(res2)
        good_set += isgood
        every_it_map_1n.append(res1)
        every_it_map_2n.append(res2)
        print('-----------------------------', i, good_set)
    clss = []
    for cls, good_num in per_cls_res.items():
        clss.append(cls)
        print(('{} = {:.4f}'.format(cls, good_num / num_tests)))
    print(('Mean AP = {:.4f}'.format(good_set / num_tests)))

    n1_5per = np.percentile(every_it_map_1n, 5, axis=0)
    n1_95per = np.percentile(every_it_map_1n, 95, axis=0)
    n2_5per = np.percentile(every_it_map_2n, 5, axis=0)
    n2_95per = np.percentile(every_it_map_2n, 95, axis=0)
    print("semi_sup")
    for i in range(len(clss)):
        print(('{} {:.1f} - {:.2f}'.format(clss[i], 100*n1_5per[i], 100*n1_95per[i])))

    print("++++++++++++++++++++++++++")
    print("only_labeled")
    for i in range(len(clss)):
        print(('{} {:.1f} - {:.2f}'.format(clss[i], 100*n2_5per[i], 100*n2_95per[i])))
    print("++++++++++++++++++++++++++")

    print("fn ", np.percentile(np.mean(every_it_map_1n, axis=1), 5),
          np.percentile(np.mean(every_it_map_1n, axis=1), 95))
    print("sn ", np.percentile(np.mean(every_it_map_2n, axis=1), 5),
          np.percentile(np.mean(every_it_map_2n, axis=1), 95))

    print()
    exit()

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if args.net == 'vgg16':
        net = vgg16()
    elif args.net == 'res50':
        net = resnetv1(num_layers=50)
    elif args.net == 'res101':
        net = resnetv1(num_layers=101)
    elif args.net == 'res152':
        net = resnetv1(num_layers=152)
    elif args.net == 'mobile':
        net = mobilenetv1()
    else:
        raise NotImplementedError

    # load model
    net.create_architecture("TEST", imdb.num_classes, tag='default',
                            anchor_scales=cfg.ANCHOR_SCALES,
                            anchor_ratios=cfg.ANCHOR_RATIOS)
    net.create_training_testing_vars()

    if args.model:
        print(('Loading model check point from {:s}').format(args.model))
        saver = tf.train.Saver()
        saver.restore(sess, args.model)
        print('Loaded.')
    else:
        print(('Loading initial weights from {:s}').format(args.weight))
        sess.run(tf.global_variables_initializer())
        print('Loaded.')
    test_net(sess, net, imdb, filename, max_per_image=args.max_per_image, annot_file_tag=wait_name)

    sess.close()
