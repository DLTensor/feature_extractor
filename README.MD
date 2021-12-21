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
#### 提取数据特征  【此功能将提取数据列表中的特征，并存放到指定目录，或者服务器上特定存放特征的路径下】
> python main.py --file ./test.txt --size 416 --weights yolov3.weights --cfg yolov3.cfg --feat --feat-type yolov3 --feat-save FEAT

file 数据路径      
size 图像尺度   
weights yolov3模型文件路径    
cfg yolov3配置文件路径    
feat 功能开关，打开开关能够实现特征提取  
feat-type 提取的特征类别
feat-save 存储特征的位置【若为空，将会存储在与输入文件【REMAP 或者 JPEGImages】相对性的特征存储目录【FEAT】下】
同时会输出不存在的文件列表，与数据路径同目录

#### 数据聚类   【此功能将根据指定的类别数目，输出聚类结果】
> python main.py --file ./test.txt --clusters 2 --feat-type yolov3 --save cluster --cluster

file 数据路径   
cluster 类别数量    
feat-type 使用的特征类别   
save 输出文件的存储路径 
同时会输出不存在的文件列表，与数据路径同目录

输出文件为：dict_filename.txt file_cluster.txt index_cluster.txt out_cluster.txt
dict_filename.txt：
为字典格式数据
存储含义： {"类别1":[图片1， 图片2], "类别2":[图片1， 图片2]} 
存储示例：
{"1":["xxx.jpg", "xxx.jpg"], "2":["xxx.jpg", "xxx.jpg"], ... ,"n":["xxx.jpg", "xxx.jpg"]}

file_cluster.txt：
按照文件列表存储
存储含义：类别 图片名称
存储示例：
1 xxx.jpg
1 xxx.jpg
5 xxx.jpg
2 xxx.jpg

index_cluster.txt:
按照文件列表顺序存储
存储格式为字典格式
存储含义：{"图片ID": 图片}
存储示例：
{"0":图片0, "1":图片1, ..., "n":图片n}

out_cluster.txt
根据文件列表存储
存储含义：类别【与文件顺序对应】
存储示例：
0
1
1
0
2
5


### 在聚类数据中随机提取一个数据  【此功能将根据存储的类别字典，在类别字典中随机抽取一张数据】
> python get_one_index.py --file weights/cluster_test/dict_filename.txt --save ./weights/cluster_test/ --outfile one_index_10000.txt

file 聚类生成的字典
save 存储路径
outfile 存储名称

输出为指定的outfile路径：
存储的为图片的路径列表

#### 过滤新增相似数据   【此功能为数据特征相似度过滤，将即将新增的数据列表和原始的数据列表进行相似度匹配，输出相似度低于阈值的文件列表】

> python main.py --filter-extend --pre-data weights\test\file.txt --ext-data weights\test\file_0906.txt --ext-thresh 0.995 --ext-save weights\test\file_extend.txt

filter-extend 功能开关
pre-data 历史数据图片列表
ext-data 新增数据图片列表
ext-thresh 过滤阈值
ext-save 存储路径

输出文件：指定的ext-save的文件路径，指定存储路径_ext.txt 指定存储路径_match.txt
指定的ext-save的文件路径：
此文件存储原始的文件列表+新增加的文件列表

指定存储路径_ext.txt
此文件只存储新增加的文件列表

指定存储路径_match.txt
此文件以字典形式存储
存储为与新增文件最相似的文件，以及相似度{"新增文件1.jpg":[最相似文件.jpg, 0.9998], "新增文件n.jpg":[最相似文件.jpg, 0.9998]}
存储示例：
{"12345.jpg":["1111.jpg", 0.998]}