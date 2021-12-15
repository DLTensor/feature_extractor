from resnet_feat_extract import get_feature, load_npy
from net_feat_extract import get_feature as get_cnn_feat
from hog_feat_extract import get_feature as get_hog_feat
from yolo_feat_extract import get_feature as get_yolo_feat
from cluster import get_kmean_clusters
from filter_func import filter_opt, get_extend_file
import argparse
import time


def get_image_feature(image_path, image_size=416, feat_type='yolov3',
                      yolo_model='./weights/indemind/yolov3.cfg',
                      weights_path='./weights/indemind/yolov3_best.weights'):
    '''
    :param image_path:
    :param image_size:
    :param feat_type: resnet50, resnet101, cnn, hog, yolov3
    :return:
    '''
    pic_dir, therd_size = image_path, image_size
    if feat_type == 'resnet50' or feat_type == 'resnet101':
        get_feature(pic_dir, therd_size, feat_type)
    elif feat_type == 'cnn':
        get_cnn_feat(pic_dir, therd_size, feat_type)
    elif feat_type == 'hog':
        get_hog_feat(pic_dir, therd_size, feat_type)
    elif feat_type == 'yolov3':
        get_yolo_feat(pic_dir, therd_size, yolo_model, weights_path)
    else:
        print('please choose feature type in [resnet50, resnet101, cnn, hog, yolov3]!!')
        return
    print('feature extract done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='./test.txt', help='image file list')
    parser.add_argument('--size', type=int, default=416, help='image size for feature extract')
    parser.add_argument('--weights', type=str, default='./weights/indemind/yolov3_best.weights', help='weights path')
    parser.add_argument('--cfg', type=str, default='./weights/indemind/yolov3.cfg', help='cfg file path')
    parser.add_argument('--feat', action='store_true', help='feat extract func')
    parser.add_argument('--feat-type', type=str, default='yolov3', help='feat extract type')
    parser.add_argument('--cluster', action='store_true', help='kmeans func')
    parser.add_argument('--save', type=str, default='./clusters', help='cluster dict file')
    parser.add_argument('--clusters', type=int, default='2', help='cluster nums')
    parser.add_argument('--filter-extend', action='store_true', help='filter extend dataset func')
    parser.add_argument('--pre-data', type=str, default='./before_npy.txt', help='before npy list')
    parser.add_argument('--ext-data', type=str, default='./extend_npy.txt', help='extend npy list')
    parser.add_argument('--ext-thresh', type=float, default='0.5', help='extend filter threshold')
    parser.add_argument('--ext-save', type=str, default='./update_extend.txt', help='cluster dict file')

    opt = parser.parse_args()

    pic_dir, therd_size = opt.file, opt.size
    yolo_model_cfg, weights_path = opt.cfg, opt.weights
    feat_type = opt.feat_type
    cluster_nums = opt.clusters

    filter_ext = opt.filter_extend
    base_npy_list = opt.pre_data
    ext_npy_list = opt.ext_data
    extend_thresh = opt.ext_thresh
    extend_file = opt.ext_save

    start_time = time.time()
    if filter_ext:
        out_ext_list, out_update_list = filter_opt(base_npy_list, ext_npy_list, extend_thresh)
        get_extend_file(out_update_list, extend_file)
    if opt.feat:
        get_image_feature(pic_dir, therd_size, feat_type)
    if opt.cluster:
        get_kmean_clusters(pic_dir, cluster_nums, opt.save, feat_type)
    if not (opt.feat or opt.cluster or opt.filter_extend):
        print('please choose one function [feat, cluster, filter-extend]')
    end_time = time.time()
    print("time cost:", end_time - start_time, '秒')


