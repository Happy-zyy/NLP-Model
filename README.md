# NLP-Model
Learning and demonstrate some  classical model



## 目录

* [Text-CNN](#text-cnn)
* [Glove](#glove)


## <span id="text-cnn">Text-CNN</span>

### 1. 数据集介绍
1.1 实验的过程中只使用了[MR数据集](https://www.cs.cornell.edu/people/pabo/movie-review-data/)，验证方式是10 folds的交叉验证方式。
>>  数据集中包含了5331 positive and 5331 negative processed sentences / snippets. Introduced in Pang/Lee ACL 2005. Released July 2005.

2.1 词向量包含以下三种(**可以任意选一种或多种累加当作一个词不同的channel**):
+ **CNN-rand**:句子中的的word vector都是随机初始化的，同时当做CNN训练过程中需要优化的参数；
+ **CNN-static**:句子中的word vector是使用word2vec预先对Google News dataset(about 100 billion words)进行训练好的词向量表中的词向量。且在CNN训练过程中作为固定的输入，不作为优化的参数；
+ **CNN-non-static**:句子中的word vector是使用word2vec预先对Google News dataset(about 100 billion words)进行训练好的词向量表中的词向量。在CNN训练过程中作为固定的输入，做为CNN训练过程中**需要优化**的参数；

说明：

> + GoogleNews-vectors-negative300.bin.gz词向量表是通过word2vec使用命令预先训练好，花费时间较长。
已经训练好的：[GoogleNews-vectors-negative300.bin.gz百度云盘下载地址](https://pan.baidu.com/share/init?surl=OglaQBBO30d5KdzZNNdRSg) 密码:18yf
> + word2vec预先训练命令如：```./word2vec -train text8(语料) -output vectors.bin(输出词向量表) -cbow(训练使用模型方式) 0 -size 48 -window 5 -negative 0 -hs 1 -sample 1e-4 -threads 20 -binary 1 -iter 100```
> + 除了使用word2vec对语料库进行预先训练外，也可以使用glove或FastText进行词向量训练。


### 2.文件介绍

2.1 **process\_data.py**：加载Google训练的词向量表GoogleNews-vectors-negative300.bin，并对文本数据做一些预处理，使其转化为NN易用的形式，并将其存储在文件中。
最终存储为一个word\_vec.p,其文件存储的内容是[**随机词向量表，已训练好的词向量表， 词频字典， 最大句子长度， revs**];
其中revs是一个结构体列表,列表中的每个元素如下所示：
```
{
"y":0/1          #标签
"num_words":int  #句子长度
"text":str       #句子
"split":[0,10]   #十折交叉使用
}
```
2.2 **text_cnn_main.py**: 主程序文件。读取以上word_vec.p文件内容，设置一些配置信息并设置一些网络运行时需要的参数。
2.3 **text_cnn_model.py**：text-cnn模型文件。


### 3.结果展示
![结果](./Text_CNN/Result/分类结果.jpg)

---
> [参考博客](https://jianwenjun.xyz/2018/03/16/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C-TextCNN-%E5%9C%A8%E5%8F%A5%E5%AD%90%E5%88%86%E7%B1%BB%E4%B8%8A%E7%9A%84%E5%AE%9E%E7%8E%B0/) 特此感谢

## <span id="glove">Glove</span>
