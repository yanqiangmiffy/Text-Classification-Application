# Text-Classification-Application
文本分类应用,采用模型是Text-CNN
- 英文数据集：影评评论
- 中文数据集(已去除停用词)：旅客酒店住宿评论；网络购物评论；书评；对话语料。
原始语料以及预处理脚本请见：链接：https://share.weiyun.com/5UoZkUx 密码：6tnnxy
- 训练好的模型：链接：https://share.weiyun.com/5tYgD3F 密码：weumvg
## 运行
英文：
![](https://github.com/yanqiangmiffy/Text-Classification-Application/blob/master/assets/result_en.png)
准训练结果：验证集76%左右

中文：
![](https://github.com/yanqiangmiffy/Text-Classification-Application/blob/master/assets/result_ch.png)
准训练结果：验证集91%左右
## 数据统计
中文
```
x.shape -> (54568, 50)
y.shape -> (54568, 2)
len(vocabulary) -> 52822
len(vocabulary_inv) -> 52822

X_train.shape -> (43654, 50)
y_train.shape -> (43654, 2)
X_test.shape -> (10914, 50)
y_test.shape -> (10914, 2)
```
英文
```
x.shape -> (10662, 56)
y.shape -> (10662, 2)
len(vocabulary) -> 12766
len(vocabulary_inv) -> 12766

X_train.shape -> (8529, 56)
y_train.shape -> (8529, 2)
X_test.shape -> (2133, 56)
y_test.shape -> (2133, 2)
```
## 资料
- [CNN-text-classification-keras](https://github.com/bhaveshoswal/CNN-text-classification-keras)
- [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
- [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
-[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
