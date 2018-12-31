from collections import defaultdict
import pandas as pd
import numpy as np
import tensorflow.contrib.keras as kr

def getWordsVect(config, W):
    word_ids = defaultdict(int)
    W_list = []
    W_list.append([0.0] * config.embedding_dim)
    count = 1
    for word,vector in W.items():
        W_list.append(vector.tolist())
        word_ids[word] = count
        count = count + 1
    return word_ids,W_list


def get_train_test_data(word_ids, data_set_df, label, sentence_max_len, cv_id=9):
    """将句子转换为id表示"""
    s = set()
    for struct in data_set_df:
        s.add(struct[label])
    cat = list(s)
    cat_to_id = dict(zip(cat, range(len(cat))))

    data_id, label_id = [], []
    for i in range(len(data_set_df)):
        data_id.append([word_ids[x] for x in data_set_df[i]['text'] if x in word_ids])
        label_id.append(cat_to_id[data_set_df[i][label]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度

    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, sentence_max_len, padding="pre")
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    train_index, test_index = [], []
    if cv_id >= 0:
        for x in range(len(data_set_df)):
            if int(data_set_df[x]["split"]) < cv_id:
                train_index.append(x)
            else:
                test_index.append(x)

        print("************")
        print("train_Num",len(train_index))
        print("test_Num", len(test_index))
        return x_pad[train_index], y_pad[train_index], x_pad[test_index], y_pad[test_index]
    else:
        return x_pad, y_pad