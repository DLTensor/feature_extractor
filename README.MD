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
####提取数据特征  
> python main.py --file ./test.txt --size 416 --weights yolov3.weights --cfg yolov3.cfg --feat --feat-type yolov3

file 数据路径      
size 图像尺度   
weights yolov3模型文件路径    
cfg yolov3配置文件路径    
feat 功能开关，打开开关能够实现特征提取  
feat-type 提取的特征类别

####数据聚类
> python main.py --file ./test.txt --clusters 2 --feat-type yolov3 --save cluster --cluster

file 数据路径   
cluster 类别数量    
feat-type 使用的特征类别   
save 输出文件的存储路径  