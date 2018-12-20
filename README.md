# NLP-Model
Learning and demonstrate some  classical model



## 目录

* [Text-CNN](#text-cnn)
* [Glove](#glove)


## <span id="text-cnn">Text-CNN</span>

### 1. 数据集介绍
1.1 实验的过程中只使用了[MR数据集](https://www.cs.cornell.edu/people/pabo/movie-review-data/)，验证方式是10 folds的交叉验证方式。
>>  数据集中包含了5331 positive and 5331 negative processed sentences / snippets.   
>Introduced in Pang/Lee ACL 2005. Released July 2005.   

---

1.2 词向量包含以下三种**（可以任意选一种或多种累加当作一个词不同的channel）**：  
**CNN-rand** : 句子中的的word vector都是随机初始化的，同时当做CNN训练过程中需要优化的参数；  
**CNN-static** : 句子中的word vector是使用word2vec预先对Google News dataset (about 100 billion words)进行训练好的词向量表中的词向量。且在CNN训练过程中作为固定的输入，不作为优化的参数;  
**CNN-non-static** : 句子中的word vector是使用word2vec预先对Google News dataset (about 100 billion words)进行训练好的词向量表中的词向量。在CNN训练过程中作为固定的输入，做为CNN训练过程中**需要优化**的参数； 



### 2.文件介绍

**process\_data.py**：加载Google训练的词向量表GoogleNews-vectors-negative300.bin，并对文本数据做一些预处理，使其转化为NN易用的形式，并将其存储在文件中。最终存储为一个word\_vec.p,其文件存储的内容是[ 随机词向量表，已训练好的词向量表， 词频字典， 最大句子长度， revs ]  
其中revs是一个列表



#### 作用

1. 修饰变量，说明该变量不可以被改变；
2. 修饰指针，分为指向常量的指针和指针常量；
3. 常量引用，经常用于形参类型，即避免了拷贝，又避免了函数对值的修改；
4. 修饰成员函数，说明该成员函数内不能修改成员变量。

#### 使用


## <span id="glove">Glove</span>
