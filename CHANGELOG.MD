#版本更改记录
## add error process
增加文件异常等错误处理机制
## add introductions
增加输出文件说明
## fix:image|npy lost
修复图片或者特征文件丢失问题；
支持指定存储目录【但存储目录下没有子目录】；
模型存储路径与数据集结构相匹配，即使用【图片路径】替换为【特征路径】
## fix: cfg&model lost
修复cfg model找不到的bug
增加score列表输出
## fix: npy empty
增加npy文件验证机制
## amend: amend file format
数据过滤输入由npy路径修订为图片路径
## V1.0.1.1版本 2021.12.14
1. 增加新增数据过滤功能【使用余弦相似度进行打分】
2. 聚类输出由npy路径更改为图片途径
## V1.0.1版本  2021.12.01
增加时间统计打印
## V1.0.0版本  2021.11.23
1. 实现特征提取功能，可提取yolov3，cnn，resnet50，resnet101，hog特征的提取与存储
2. 实现kmeans聚类功能，可根据1中的特征进行聚类运算，输出不同簇的文件列表
