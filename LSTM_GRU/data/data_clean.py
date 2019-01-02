
# coding: utf-8

# In[10]:


import pickle
import numpy as np
from collections import defaultdict,OrderedDict
import re
from tqdm import tqdm
import pandas as pd
from bitarray import bitarray


# In[185]:


def clean_string(string,TREC=False):
    string = re.sub(r"[^A-Za-z0-9,!?.]", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"(?<=\s)\w(?=\s)", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


# In[245]:


def load_data_k_cv(folder,cv=10,miniData = True): 
    """struct : text
                device
                brand
                model
                split
       word_cab : 词频字典
    
    """
    word_cab=defaultdict(int)
    df = []
    num = 0
    with open(folder,'rb') as f:
        for line in tqdm(f):
            line = line.decode(encoding='ISO-8859-1')
            row = list(map(lambda x : x.strip(),line.strip().split("|")))[1:]
            if not (5 <= len(row) <= 6) :
                continue
            row = row[:3] + row[3].split(",") + row[4:]       
            if len(row) != 6:
                continue
            row = list(map(lambda x : clean_string(x), row))
            row.append(np.random.randint(0, cv))
            df.append({"text":str(row[5]) +" "+ row[0]+" " + row[1],"device":row[2],"brand":row[3],"model":row[4],"split":row[6]})
            num += 1
            if miniData and num == 10000:
                break

    word_cab = defaultdict(int)
    sentence_max_len = 0
    final_df = []
    
    print("cleaning data")
    for struct in tqdm(df):
        length = len(struct["text"].split())
        if length <= 200:
            struct["text"] = clean_string(struct["text"])
            sentence_max_len = max(sentence_max_len, len(struct["text"].split()))
            final_df.append(struct)
            for word in struct["text"].split():
                word_cab[word] += 1
    print("cleaning data finish!")
    return final_df, word_cab, sentence_max_len


# In[246]:


def load_binary_vec(fname, vocab):
    word_vecs = {}
    with open(fname, 'rb') as fin:
        header = fin.readline()
        vocab_size, vector_size = list(map(int, header.split()))
        binary_len = np.dtype(np.float32).itemsize * vector_size
        # vectors = []
        for i in tqdm(range(vocab_size)):
            # read word
            word = b''
            while True:
                ch = fin.read(1)
                if ch == b' ':
                    break
                word += ch
            # print(str(word))
            word = word.decode(encoding='ISO-8859-1')
            if word in vocab:
                word_vecs[word] = np.fromstring(fin.read(binary_len), dtype=np.float32)
            else:
                fin.read(binary_len)
            fin.read(1)  # newline
        return word_vecs


# In[247]:


def add_unexist_word_vec(word_vecs, word_cab):
    for word in tqdm(set(word_cab.keys() -word_vecs.keys())):
        word_vecs[word] = np.random.uniform(-0.1,0.1,100)


# In[248]:


data_folder = r"all.txt"
w2v_file = r'vec1.bin'


# In[265]:


print("load text")
df, word_cab, sentence_max_len = load_data_k_cv(data_folder, 10, False)
print("finish text load !!!")


# In[266]:


brandCount = defaultdict(int)
for struct in df:
    brandCount[(struct['brand'])] += 1


# In[267]:


usefulBrand = set()
for k, v in brandCount.items():
    if v > 50:
        usefulBrand.add(k)


# In[268]:


for i in range(len(df)-1,-1,-1):
    if df[i]['brand'] not in usefulBrand:
        df.pop(i)


# In[271]:


len(df)


# In[282]:


with open("banner.txt","wb") as f:
    for struct in df:
        f.write(bytes(struct['text']+'\n', encoding="utf8"))
    


# In[283]:


print("load word2vec")
word_vecs = load_binary_vec(w2v_file, word_cab)
print("finish word2vec load !!!")


# In[285]:


add_unexist_word_vec(word_vecs,word_cab)


# In[286]:


len(word_vecs)


# In[287]:


pickle.dump([df,word_vecs,word_cab,sentence_max_len],open(r'word_vec.p','wb'))


