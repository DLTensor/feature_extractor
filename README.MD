# 数据过滤工具包 2021.11.23
##安装包
pytorch     
numpy   
sklearn     
skimage  
imgaug  
json    
cv2     
pillow  
matplotlib

## 使用说明
#### 提取数据特征  
> python main.py --file ./test.txt --size 416 --weights yolov3.weights --cfg yolov3.cfg --feat --feat-type yolov3

file 数据路径      
size 图像尺度   
weights yolov3模型文件路径    
cfg yolov3配置文件路径    
feat 功能开关，打开开关能够实现特征提取  
feat-type 提取的特征类别

#### 数据聚类
> python main.py --file ./test.txt --clusters 2 --feat-type yolov3 --save cluster --cluster

file 数据路径   
cluster 类别数量    
feat-type 使用的特征类别   
save 输出文件的存储路径 
同时会输出不存在的文件列表，与数据路径同目录

### 在聚类数据中随机提取一个数据
> python get_one_index.py --file weights/cluster_test/dict_filename.txt --save ./weights/cluster_test/ --outfile one_index_10000.txt

file 聚类生成的字典
save 存储路径
outfile 存储名称
同时会输出不存在的文件列表，与数据路径同目录

#### 过滤新增相似数据

> python main.py --filter-extend --pre-data weights\test\file.txt --ext-data weights\test\file_0906.txt --ext-thresh 0.995 --ext-save weights\test\file_extend.txt

filter-extend 功能开关
pre-data 历史数据图片列表
ext-data 新增数据图片列表
ext-thresh 过滤阈值
ext-save 存储路径