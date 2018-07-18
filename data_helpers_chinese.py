import numpy as np
import itertools
from collections import Counter
import jieba
def load_data_and_labels():
    """
    生成数据和标签
    :return:
    """
    # 从文件加载数据
    positive_examples=list(open('data/chinese/rt-polarity.pos','r',encoding='utf-8').readlines())
    positive_examples=[s.strip() for s in positive_examples] # 去除句子前后的空格
    negative_examples=list(open('data/chinese/rt-polarity.neg','r',encoding='utf-8').readlines())
    negative_examples=[s.strip() for s in negative_examples] # 去除句子前后的空格

    # 分割词
    x_text=positive_examples+negative_examples
    x_text=[sent.split(" ") for sent in x_text]

    # 生成标签
    positive_labels=[[0,1] for _ in positive_examples]
    negative_labels=[[1,0] for _ in negative_examples]
    y=np.concatenate([positive_labels,negative_labels],0)
    return [x_text,y]


def pad_sentences(sentences,padding_word="<PAD/>",sequence_length=50):
    """
    填充句子，使所有句子句子长度等于最大句子长度
    :param sentences: 
    :param padding_word: 
    :return: padded_sentences
    """
    # import pandas as pd
    # length=pd.DataFrame([len(x) for x in sentences])
    # print(length.describe())
    # print(sequence_length)
    padded_sentences=[]
    for i in range(len(sentences)):
        sentence=sentences[i]
        if sequence_length>len(sentence):
            num_padding=sequence_length-len(sentence)
            new_sentence=sentence+num_padding*[padding_word]
            padded_sentences.append(new_sentence)
        else:
            padded_sentences.append(sentence[:sequence_length])
    return padded_sentences


def build_vocab(sentences):
    """
    index和word创建
    :param sentences:
    :return:
    """
    # 创建词汇表
    word_counts=Counter(itertools.chain(*sentences))
    # 将index映射到word
    vocabulary_inv=[x[0] for x in word_counts.most_common()]
    vocabulary_inv=list(sorted(vocabulary_inv))
    # 将word映射到index
    vocabulary={x:i for i,x  in enumerate(vocabulary_inv)}
    return [vocabulary,vocabulary_inv]


def build_input(sentences,labels,vocabulary):
    x=np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y=np.array(labels)
    return [x,y]


def build_input_chinese(input_x):
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)

    sequence_length=50
    padding_word = "<PAD/>"
    input_x=' '.join(jieba.cut(input_x))
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
    vocabulary,vocabulary_inv=build_vocab(sentences_padded)
    x,y=build_input(sentences_padded,labels,vocabulary)
    return [x,y,vocabulary,vocabulary_inv]


# if __name__ == '__main__':
    # x=build_predict_input('服务 招待所 水平 网络 不好 值得 推荐')
    # x, y, vocabulary, vocabulary_inv=load_data()
    # print(x.shape)