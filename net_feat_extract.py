# !/usr/bin/python
# coding: utf8
# @Time    : 2021-11-22
# @Author  : LYK
# @Software: PyCharm

from torch import nn
from resnet_feat_extract import *

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 25, kernel_size=3),
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=3),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(520200, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_feature(pic_dir, therd_size, model_name='cnn'):
    transform = transforms.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CNN().to(device)
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
            print(feature_array)

            dst, basename = os.path.split(image_file)
            dst_npy = os.path.join(dst, model_name + basename[:-3] + 'npy')
            np.save(dst_npy, feature_array)
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
        print(feature_array)

        dst, basename = os.path.split(pic_dir)
        dst_npy = os.path.join(dst, model_name + basename[:-3] + 'npy')
        np.save(dst_npy, feature_array)


if __name__ == '__main__':
    pic_dir, therd_size, model_name = './images/14_1614067694109.jpg', 416, 'resnet50'
    get_feature(pic_dir, therd_size, model_name)