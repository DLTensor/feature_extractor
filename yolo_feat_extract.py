# !/usr/bin/python
# coding: utf8
# @Time    : 2021-11-23
# @Author  : LYK
# @Software: PyCharm

from yolo_model import load_model
from utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
from utils.datasets import ImageFolder
from utils.transforms import Resize, DEFAULT_TRANSFORMS
from utils_pakege import *
import cv2


def detect_image(model, image, img_size=416, conf_thres=0.5, nms_thres=0.5):
    """Inferences one image with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param image: Image to inference
    :type image: nd.array
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: Detections on image with each detection in the format: [x1, y1, x2, y2, confidence, class]
    :rtype: nd.array
    """
    model.eval()  # Set model to evaluation mode

    # Configure input
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])(
            (image, np.zeros((1, 5))))[0].unsqueeze(0)

    if torch.cuda.is_available():
        input_img = input_img.to("cuda")

    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        # detections = non_max_suppression(detections, conf_thres, nms_thres)
        # detections = rescale_boxes(detections[0], img_size, image.shape[:2])
    return detections.cpu()


def get_feature(pic_dir, therd_size, model_path, weights_path, save_path=''):
    model = load_model(model_path, weights_path)
    model_name = os.path.basename(model_path)[:-4]
    if 'txt' in os.path.basename(pic_dir):
        pic_list = []
        with open(pic_dir, 'r') as file_list:
            lines = file_list.readlines()
            for line in lines:
                value = line.strip()
                pic_list.append(value)
        issue_files = []
        root_file, _ = os.path.split(pic_dir)
        for image_file in tqdm(pic_list):
            if not os.path.exists(image_file):
                issue_files.append(image_file)
                continue
            if cv2.imread(image_file) is None:
                print(image_file)
                continue
            img = skimage.io.imread(image_file)
            if len(img.shape) == 2:
                image = np.expand_dims(img, axis=2)
                img = np.concatenate((image, image, image), axis=-1)

            feats = detect_image(model, img, therd_size)
            feature_array = feats.data.numpy()

            dst_split, basename = os.path.split(image_file)
            if save_path == '':
                dst = dst_split.replace('REMAP', 'FEAT')
            else:
                dst = save_path
            if not os.path.exists(dst):
                os.makedirs(dst)
            dst_npy = os.path.join(dst, model_name + '_' + basename[:-3] + 'npy')
            if not os.path.exists(dst_npy):
                np.save(dst_npy, feature_array)
        if len(issue_files) > 0:
            issue_files_path = os.path.join(root_file, 'error_images.txt')
            file = open(issue_files_path, 'w')
            for index, issue_file in enumerate(issue_files):
                file.write(str(issue_file) + '\n')
            file.close()
            print("save not exists images files in:", issue_files_path)
        return
    else:
        img = skimage.io.imread(pic_dir)
        feats = detect_image(model, img, therd_size)
        feature_array = feats.data.numpy()

        dst, basename = os.path.split(pic_dir)
        dst_npy = os.path.join(dst, model_name + '_' + basename[:-3] + 'npy')
        np.save(dst_npy, feature_array)
        return


if __name__ == '__main__':
    pic_dir, therd_size, model_name, weights_path = './test.txt', 416, './weights/indemind/yolov3.cfg', './weights/indemind/yolov3_best.weights'
    get_feature(pic_dir, therd_size, model_name, weights_path)
    print('done')
