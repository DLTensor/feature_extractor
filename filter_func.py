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
from cluster import save_cluster as save_match_max
from utils_pakege import np


def filter_opt(base_list_file, ext_list_file, extend_thresh, debug=False):
    base_npy_list, ext_npy_list, file_base_list, file_ext_list = get_input_list(base_list_file, ext_list_file)
    base_features = []
    ext_files = []
    base_files = file_base_list.copy()
    match_max = {}

    for index, file in tqdm(enumerate(base_npy_list), total=len(base_npy_list)):
        if not os.path.exists(file):
            continue
        npy_array = load_npy(file)
        pca_array = pca_process(npy_array, n_dims=351)
        # pca_array = npy_array.copy()
        flat_pca_aray = np.array(pca_array).flatten().tolist()
        base_features.append(flat_pca_aray)

    if len(base_features) == 0:
        print('base_features is empty!')
        return [], [], []

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
        score_list = score_array.tolist()
        max_score = score_array.max()
        max_index = score_list.index(max(score_list))
        match_value = [base_files[max_index], max_score]
        match_max[file_ext_list[index_ext]] = match_value
        min_score = score_array.min()
        # print(max_score, min_score)
        if max_score < extend_thresh:
            if not debug:
                np_base_features = np.concatenate((np_base_features, tmp_flat))
                base_files.append(file_ext_list[index_ext])
            ext_files.append(file_ext_list[index_ext])
    return ext_files, base_files, match_max


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
    file_base_list = []
    npy_ext_list = []
    file_ext_list = []
    if not os.path.exists(base_file) or not os.path.exists(ext_file):
        print('your input files not exists!')
        return
    issue_base_files = []
    issue_ext_files = []
    root_base_file, _ = os.path.split(base_file)
    root_ext_file, _ = os.path.split(ext_file)
    with open(base_file, 'r') as npy_file:
        lines = npy_file.readlines()
        for line in lines:
            value = line.strip()
            if not os.path.exists(value):
                issue_base_files.append(value)
                continue
            dirname, basename = os.path.split(value)
            dirname = dirname.replace('REMAP', 'FEAT')
            dirname = dirname.replace('JPEGImages', 'FEAT')
            if not basename[-3:] == 'npy':
                basename = basename[:-3] + 'npy'
            feat_name = 'yolov3' + '_' + basename
            feat_path = os.path.join(dirname, feat_name)
            file_base_list.append(value)
            npy_base_list.append(feat_path)
    with open(ext_file, 'r') as npy_file:
        lines = npy_file.readlines()
        for line in lines:
            value = line.strip()
            if not os.path.exists(value):
                issue_ext_files.append(value)
                continue
            file_ext_list.append(value)
            dirname, basename = os.path.split(value)
            dirname = dirname.replace('REMAP', 'FEAT')
            dirname = dirname.replace('JPEGImages', 'FEAT')
            if not basename[-3:] == 'npy':
                basename = basename[:-3] + 'npy'
            feat_name = 'yolov3' + '_' + basename
            feat_path = os.path.join(dirname, feat_name)
            npy_ext_list.append(feat_path)

    if len(issue_base_files) > 0:
        issue_files_path = os.path.join(root_base_file, 'error_base_file.txt')
        file = open(issue_files_path, 'w')
        for index, issue_file in enumerate(issue_base_files):
            file.write(str(issue_file) + '\n')
        file.close()
        print("save not exists base files in:", issue_files_path)
    if len(issue_ext_files) > 0:
        issue_files_path = os.path.join(root_ext_file, 'error_ext_file.txt')
        file = open(issue_files_path, 'w')
        for index, issue_file in enumerate(issue_ext_files):
            file.write(str(issue_file) + '\n')
        file.close()
        print("save not exists ext files in:", issue_files_path)
    return npy_base_list, npy_ext_list, file_base_list, file_ext_list


if __name__ == '__main__':
    path = 'E:/dataset/REMAP/TEST/ICE_20210830/data-9/gray/cam0/'
    path_0906 = 'E:/dataset/REMAP/TEST/ICE_20210906/data-10/gray/cam0/'
    out_ext_list, out_update_list, out_match_max = filter_opt('weights/test/file.txt', 'weights/test/file_0906.txt', 0.995, debug=True)
    print(out_match_max)
    # get_extend_file(out_update_list, 'weights/test/file_extend.txt')
    # print(out_ext_list)
    save_match_max('weights/test/file_match_case', out_match_max)

