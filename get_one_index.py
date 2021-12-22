import os
from cluster import load_cluster
import argparse
import random


def extract_one_path(input_file_name, save_path, name):
    path = input_file_name
    if not os.path.exists(path):
        print('your input file not exists!')
        return [], []

    cluster_path = os.path.join(save_path, name)
    save_file = open(cluster_path, 'w')

    dic_files = load_cluster(path)
    print(dic_files)
    images = []
    file_list = []
    for file_index in dic_files:
        random_index =  random.randint(0, len(dic_files[file_index]) - 1)
        file_name = dic_files[file_index][random_index]
        file_list.append(file_name)
        save_file.write(str(file_name) + '\n')
        images.append(file_name)
    save_file.close()
    return images, file_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='./weights/cluster_test/', help='output file root path')
    parser.add_argument('--file', type=str, default='weights/cluster_test/dict_filename.txt', help='image file list')
    parser.add_argument('--outfile', type=str, default='one_index_10000.txt', help='output file list')
    opt = parser.parse_args()

    save_root_path = opt.save
    output_filename = opt.outfile
    dic_path = opt.file
    _, _ = extract_one_path(dic_path, save_root_path, output_filename)