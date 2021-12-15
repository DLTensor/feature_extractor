"""
# !/usr/bin/python
# coding: utf8
# @Time    : 2021-12-14
# @Author  : LYK
# @Software: PyCharm
"""
from utils_pakege import tqdm
from utils_pakege import os
from cluster import load_npy, pca_process
from utils_pakege import np


def filter_opt(base_list_file, ext_list_file, extend_thresh):
    base_npy_list, ext_npy_list = get_input_list(base_list_file, ext_list_file)
    base_features = []
    ext_files = []
    base_files = base_npy_list.copy()

    for index, file in tqdm(enumerate(base_npy_list), total=len(base_npy_list)):
        if not os.path.exists(file):
            continue
        npy_array = load_npy(file)
        pca_array = pca_process(npy_array, n_dims=351)
        # pca_array = npy_array.copy()
        flat_pca_aray = np.array(pca_array).flatten().tolist()
        base_features.append(flat_pca_aray)

    np_base_features = np.array(base_features)
    for index_ext, ext_file in tqdm(enumerate(ext_npy_list), total=len(ext_npy_list)):
        if not os.path.exists(ext_file):
            continue
        index_length, _ = np_base_features.shape
        ext_npy_array = load_npy(ext_file)
        # ext_pca_array = ext_npy_array.copy()
        ext_pca_array = pca_process(ext_npy_array, n_dims=351)
        ext_flat_pca_array = np.array(ext_pca_array).flatten().tolist()
        tmp_flat = np.array([ext_flat_pca_array])
        muti_tmp_flat = np.tile(tmp_flat, (index_length, 1))

        dist_cos, score_cos = cosine_distance(muti_tmp_flat, np_base_features)
        score_array = np.diagonal(score_cos)
        max_score = score_array.max()
        min_score = score_array.min()
        # print(max_score, min_score)
        if max_score < extend_thresh:
            np_base_features = np.concatenate((np_base_features, tmp_flat))
            ext_files.append(ext_file)
            base_files.append(ext_file)
    return ext_files, base_files


def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    dist = 1. - similiarity
    return dist, similiarity


def get_extend_file(update_list, save_filename):
    file = open(save_filename, 'w')
    for index, update_file in enumerate(update_list):
        file.write(str(update_file) + '\n')
    file.close()


def get_input_list(base_file, ext_file):
    npy_base_list = []
    npy_ext_list = []
    if not os.path.exists(base_file) or not os.path.exists(ext_file):
        print('your input files not exists!')
        return
    with open(base_file, 'r') as npy_file:
        lines = npy_file.readlines()
        for line in lines:
            value = line.strip()
            npy_base_list.append(value)
    with open(ext_file, 'r') as npy_file:
        lines = npy_file.readlines()
        for line in lines:
            value = line.strip()
            npy_ext_list.append(value)
    return npy_base_list, npy_ext_list


if __name__ == '__main__':
    path = 'E:/dataset/REMAP/TEST/ICE_20210830/data-9/gray/cam0/'
    path_0906 = 'E:/dataset/REMAP/TEST/ICE_20210906/data-10/gray/cam0/'
    out_ext_list, out_update_list = filter_opt('weights/test/npy.txt', 'weights/test/npy_0906.txt', 0.995)
    get_extend_file(out_update_list, 'weights/test/extend.txt')
    print(out_ext_list)

