### 1. 代码环境
|名称|版本|
|:-:|:-:|
|OS|ubuntu 20.04 lts|
|GPU|2080Ti|
|nvidia driver|460.84|
|cuda|11.2|
|pytorch|1.9.1|
|conda|4.10.3|

### 2. 解决方案及算法介绍
按照类型切分数据集，使用脚本为`data/code/dataset_script.py`。

对于相同类型的句子，从train集中选出最相近的两个句子做拼接，使用脚本为`data/code/cosine_similiarity.py`。

将拼接后的train集按照7:3分出dev集，使用脚本为`data/code/divide.py`。

将各类数据集合并，形成`train.txt``dev.txt``test.txt`，使用脚本为`get_whole.py`。

以上是对数据集的处理。

在模型训练部分，使用p-tuning的方法构造prompt并替换BERT的embedding层，参见`data/code/BERT/model/`文件夹。

默认训练参数详见`data/code/BERT/train.py`。