# CFEP

# 原始数据

* data文件夹，包含2015-2022三个地区EG，IR，IS发生事件的csv文件以及对应的事件类型编码本CAMEO
* 以EG数据集为例，EG.csv中每一个记录即为一条边，实体映射编码为entity2id.txt文件，eventCode映射name为dict_id2ont.json。docs_title_paragraph.json与md5_list.json每一行一一对应，即md5_list.json中的每一行md5加密的网址对应的标题与文章内容

# 数据处理

* `python data2id.py`将原始数据中每个地区csv文件的actorname转化为id，规范化EventCode
* `python split_data.py` 划分数据集，划分为train,valid,calib_train,calib_valid,test，比例为5:1:1:1:2
* `python generate_data_embedding.py`将新闻文本输入预训练语言模型（bge_base_en_v1.5），得到文本embedding
* `python get_digraphs_with_embedding.py` 每个时刻构建事件图

# 模型训练

* 划分数据集为训练集，校验训练集，校验测试集，测试集
* 训练集上训练模型，校验训练集上训练cp模型，校验测试集上计算阈值，测试集上输出覆盖率
