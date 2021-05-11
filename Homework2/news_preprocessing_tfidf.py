######################################################################
# Author : 1018512015-张翼鹏
# Date : 2021/04/29

######################################################################
# 调库区
import os
import csv
import nltk
import numpy as np
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from rich.progress import track

######################################################################

######################################################################
# 常量区
# TF-IDF算法中新闻各项所占权重
WEIGHT_OF_CATEGORY = 4
WEIGHT_OF_SUBCATEGORY = 3
WEIGHT_OF_TITLE = 2
WEIGHT_OF_ABSTRACT = 1

# 朴素算法中新闻各项所占权重
WEIGHT_OF_NOUN = 3
WEIGHT_OF_VERB = 2

######################################################################

######################################################################
# 函数区

# 读入一个字符串，返回分词后的字符串
def text_preprocessing(row_text):
    token_words = nltk.word_tokenize(row_text.lower())
    characters = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-', '...',
                  '^', '{', '}']
    # 分词、删除标点并小写
    token_words = [str for str in token_words if str not in characters]
    token_words = pos_tag(token_words)  # 词性标注
    # 词型还原
    words_lematizer = []
    wordnet_lematizer = WordNetLemmatizer()
    for word, tag in token_words:
        if tag.startswith('NN'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='n')  # n代表名词
            for i in range(WEIGHT_OF_NOUN-1):
                words_lematizer.append(word_lematizer)
        elif tag.startswith('VB'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='v')  # v代表动词
            for i in range(WEIGHT_OF_VERB-1):
                words_lematizer.append(word_lematizer)
        elif tag.startswith('JJ'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='a')  # a代表形容词
        elif tag.startswith('R'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='r')  # r代表代词
        else:
            word_lematizer = wordnet_lematizer.lemmatize(word)
        words_lematizer.append(word_lematizer)

    words_list = [word for word in words_lematizer if word not in stopwords.words('english')]
    return ' '.join(words_list)+' '
######################################################################
# tsv:[id, 大分类，二分类，标题，摘要]
# news_id_dict ->map<string,int>
# mews_dict -><int,string> string = 大分类*大分类权重+二分类*二分类权重+标题*标题权重+摘要（分词、词性还原）*摘要权重
# ["今天天气好"，"今天星期二"] ->corpus:[今天，天气，好，星期二]
# tdidf[i][j]文本i里j词的权重

######################################################################
# 功能区
# 若用该方法处理的新闻字典已经存在则不再重复操作
if not os.path.exists("news_dict.npy"):
    r_news_tsv = 'train/train_news.tsv'
    news_id_dict = {}
    news_dict = {}                                                          # 新闻字典
    new_corpus = []
    total_news = len(open(r_news_tsv, 'r',encoding='UTF-8').readlines())-1  # 新闻总数
    with open(r_news_tsv,'r',encoding='UTF-8') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')
        news_tsv_labels = tsv_reader.__next__()                             # 先把列标题读走
        for news_index in track(range(total_news),description='新闻数据处理中...'):
            record = tsv_reader.__next__()
            news_id_dict[record[0]] = news_index
            news_dict[news_index] = text_preprocessing(record[1]) * WEIGHT_OF_CATEGORY \
                                  + text_preprocessing(record[2]) * WEIGHT_OF_SUBCATEGORY \
                                  + text_preprocessing(record[3]) * WEIGHT_OF_TITLE \
                                  + text_preprocessing(record[4]) * WEIGHT_OF_ABSTRACT
            new_corpus.append(news_dict[news_index])


    np.save('news_dict.npy', news_dict)                                     # 保存新闻字典
    np.save('news_id_dict.npy', news_id_dict)                               # 保存新闻字典
    new_corpus = np.array(new_corpus)                                       # 保存新闻语料库
    np.save('new_corpus.npy',new_corpus)

######################################################################
# 测试区
#read_id_dict = np.load('news_id_dict.npy', allow_pickle=True).item()
#read_dict = np.load('news_dict.npy', allow_pickle=True).item()
#read_corpus = np.load('new_corpus.npy')
#read_corpus = read_corpus.tolist()

#vectorizer = CountVectorizer()
#transformer = TfidfTransformer()
#news_tfidf = transformer.fit_transform(vectorizer.fit_transform(read_corpus))

