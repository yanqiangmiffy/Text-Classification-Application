import numpy as np
import re
import itertools
from collections import Counter
import pickle
import os
def clean_str(string):
    """
    数据预处理
    :param string:
    :return:
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string) # 去除数字
    # 标点符号处理
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    生成数据和标签
    :return:
    """
    # 从文件加载数据
    positive_examples=list(open('data/english/rt-polarity.pos','r',encoding='latin-1').readlines())
    positive_examples=[s.strip() for s in positive_examples] # 去除句子前后的空格
    negative_examples=list(open('data/english/rt-polarity.neg','r',encoding='latin-1').readlines())
    negative_examples=[s.strip() for s in negative_examples] # 去除句子前后的空格

    # 分割词
    x_text=positive_examples+negative_examples
    x_text=[clean_str(sent) for sent in x_text]
    x_text=[sent.split(" ") for sent in x_text]

    # 生成标签
    positive_labels=[[0,1] for _ in positive_examples]
    negative_labels=[[1,0] for _ in negative_examples]
    y=np.concatenate([positive_labels,negative_labels],0)
    return [x_text,y]


def pad_sentences(sentences,padding_word="<PAD/>"):
    """
    填充句子，使所有句子句子长度等于最大句子长度
    :param sentences: 
    :param padding_word: 
    :return: padded_sentences
    """
    sequence_length=max(len(x) for x in sentences)
    padded_sentences=[]
    for i in range(len(sentences)):
        sentence=sentences[i]
        num_padding=sequence_length-len(sentence)
        new_sentence=sentence+num_padding*[padding_word]
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab():
    """
    index和word创建
    :param sentences:
    :return:
    """
    # 加载已经保存的词汇文件
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)

    voab_dir = 'data/english/vocab.pkl'
    if os.path.exists(voab_dir):
        with open(voab_dir, 'rb') as in_data:
            vocabulary = pickle.load(in_data)
            vocabulary_inv = pickle.load(in_data)
        return [vocabulary, vocabulary_inv]

    # 创建词汇表
    word_counts=Counter(itertools.chain(*sentences_padded))
    # 将index映射到word
    vocabulary_inv=[x[0] for x in word_counts.most_common()]
    vocabulary_inv=list(sorted(vocabulary_inv))
    # 将word映射到index
    vocabulary={x:i for i,x  in enumerate(vocabulary_inv)}

    with open(voab_dir,'wb') as out_data:
        print("正在保存英文文词汇表")
        pickle.dump(vocabulary,out_data,pickle.HIGHEST_PROTOCOL)
        pickle.dump(vocabulary_inv,out_data,pickle.HIGHEST_PROTOCOL)
    return [vocabulary,vocabulary_inv]


def build_input(sentences,labels,vocabulary):
    x=np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y=np.array(labels)
    return [x,y]


def build_input_english(input_x):
    vocabulary, vocabulary_inv = build_vocab()

    sequence_length=56
    padding_word = "<PAD/>"
    input_x = input_x.split(' ')
    num_padding = sequence_length - len(input_x)
    input_x = input_x + num_padding * [padding_word]
    input_x = np.array([vocabulary[word] if word in vocabulary else vocabulary[padding_word] for word in input_x])
    return input_x.reshape(1,sequence_length)


def load_data():
    """
    加载预处理后的数据
    :return:
    """
    sentences,labels=load_data_and_labels()
    sentences_padded=pad_sentences(sentences)
    vocabulary,vocabulary_inv=build_vocab()
    x,y=build_input(sentences_padded,labels,vocabulary)
    return [x,y,vocabulary,vocabulary_inv]

#
# if __name__ == '__main__':
#     x=build_input_english('i love you')
#     print(x)