# !/usr/bin/python
# coding: utf8
# @Time    : 2021-11-22
# @Author  : LYK
# @Software: PyCharm

from utils_pakege import *


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)

            x = module(x)
            print(name)
            if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
                outputs[name] = x

        return outputs


def get_picture(pic_name, transform, size):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (size, size))
    img = np.asarray(img, dtype=np.float32)
    if len(img.shape) == 2:
        # if img.shape
        image = np.expand_dims(img, axis=2)
        image = np.concatenate((image, image, image), axis=-1)
        return transform(image)
    return transform(img)


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_feature(pic_dir, therd_size, model_name):
    transform = transforms.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if 'resnet50' == model_name:
        net = models.resnet50().to(device)
        net.load_state_dict(torch.load('./weights/resnet50-19c8e357.pth'))
    elif 'resnet101' == model_name:
        net = models.resnet101().to(device)
        net.load_state_dict(torch.load('./weights/resnet101-5d3b4d8f.pth'))
    else:
        print('please choose resnet50 or resnet101')
        return
    exact_list = None
    myexactor = FeatureExtractor(net, exact_list)
    if 'txt' in os.path.basename(pic_dir):
        pic_list = []
        with open(pic_dir, 'r') as file_list:
            lines = file_list.readlines()
            for line in lines:
                value = line.strip()
                pic_list.append(value)

        for image_file in tqdm(pic_list):
            img = get_picture(image_file, transform, therd_size)
            # 插入维度
            img = img.unsqueeze(0)
            img = img.to(device)
            outs = myexactor(img)
            fc = outs['fc']
            features = fc[0]
            features = features.cpu()
            feature_array = features.data.numpy()

            dst, basename = os.path.split(image_file)
            dst_npy = os.path.join(dst, model_name + '_' + basename[:-3] + 'npy')
            np.save(dst_npy, feature_array)

        return
    else:
        img = get_picture(pic_dir, transform, therd_size)
        # 插入维度
        img = img.unsqueeze(0)
        img = img.to(device)
        outs = myexactor(img)
        fc = outs['fc']
        features = fc[0]
        features = features.cpu()
        feature_array = features.data.numpy()

        dst, basename = os.path.split(pic_dir)
        dst_npy = os.path.join(dst, model_name + '_' + basename[:-3] + 'npy')
        np.save(dst_npy, feature_array)


def load_npy(npy):
    npy_value = np.load(npy)
    return npy_value
