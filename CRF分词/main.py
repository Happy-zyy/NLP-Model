
# coding: utf-8

# In[81]:


import re
import numpy as np
from 


# In[89]:


sents = open(r'H:\分词数据\training\pku_training.utf8',encoding='utf-8').read()
sents = sents.strip()
sents = sents.split('\n') # 这个语料的换行符是\r\n


# In[105]:


sents = [re.split(' +', s) for s in sents] # 词之间以空格隔开
sents = [[w for w in s if w] for s in sents] # 去掉空字符串
np.random.shuffle(sents) # 打乱语料，以便后面划分验证集


# In[91]:


chars = {} # 统计字表
for s in sents:
    for c in ''.join(s):
        if c in chars:
            chars[c] += 1
        else:
            chars[c] = 1

min_count = 2 # 过滤低频字
chars = {i:j for i,j in chars.items() if j >= min_count} # 过滤低频字  低频字的id是0
id2char = {i+1:j for i,j in enumerate(chars)} # id到字的映射
char2id = {j:i for i,j in id2char.items()} # 字到id的映射

id2tag = {0:'s', 1:'b', 2:'m', 3:'e'} # 标签（sbme）与id之间的映射
tag2id = {j:i for i,j in id2tag.items()}

train_sents = sents[:-5000] # 留下5000个句子做验证，剩下的都用来训练
valid_sents = sents[-5000:]


# In[97]:


batch_size = 128


# In[98]:


train_sents[0]


# In[123]:


def train_generator(): #定义数据生成器
    X, Y = [], []
    while True:
        for i,text in enumerate(train_sents):
            sx,sy = [], []
            for s in text:
                sx.extend([char2id.get(c,0) for c in s])
                if len(s) == 1:
                    sy.append(0)
                elif len(s) == 2:
                    sy.extend([1,3])
                else:
                    sy.extend([1] + [2]*(len(s) - 2) + [3])
            X.append(sx)
            Y.append(sy)
            if len(X) == batch_size or i == len(train_sents)-1:
                maxlen = max([len(t) for t in X])
                X = [x+[4]*(maxlen-len(x)) for x in X]
                Y = [y+[4]*(maxlen-len(y)) for y in Y]
                yield np.array(X), to_categorical(Y, 5)
                X, Y = [], []

