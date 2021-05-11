######################################################################
# Author : 1018512015-张翼鹏
# Date : 2021/05/03

######################################################################
# 调库区
import os
import csv
import nltk
import math
import numpy as np
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from rich.progress import track
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
######################################################################

######################################################################
# 常量区
# TF-IDF算法中新闻各项所占权重
WEIGHT_OF_CATEGORY = 4
WEIGHT_OF_SUBCATEGORY = 3
WEIGHT_OF_TITLE = 2
WEIGHT_OF_ABSTRACT = 1

# TF-IDF算法中新闻各项所占权重
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

# 计算用户偏好向量
def get_user_pref(history):                     # history为用户浏览历史列表
    total_history = len(history)
    if(total_history == 0):
        return []
    history_sum = tfidf[history[0]]
    for i in range(total_history-1):
        history_sum += tfidf[history[i+1]]
    return history_sum/total_history
######################################################################

######################################################################
# 功能区

r_news_tsv = 'test/test_news.tsv'
news_id_dict = {}
news_dict = {}                                                          # 新闻字典
new_corpus = []
total_news = len(open(r_news_tsv, 'r',encoding='UTF-8').readlines())-1  # 新闻总数
with open(r_news_tsv,'r',encoding='UTF-8') as tsv_in:
    tsv_reader = csv.reader(tsv_in, delimiter='\t')
    news_tsv_labels = tsv_reader.__next__()                             # 先把列标题读走
    for news_index in track(range(total_news),description='模型训练中...'):
        record = tsv_reader.__next__()
        news_id_dict[record[0]] = news_index
        news_dict[news_index] = text_preprocessing(record[1]) * WEIGHT_OF_CATEGORY \
                              + text_preprocessing(record[2]) * WEIGHT_OF_SUBCATEGORY \
                              + text_preprocessing(record[3]) * WEIGHT_OF_TITLE \
                              + text_preprocessing(record[4]) * WEIGHT_OF_ABSTRACT
        new_corpus.append(news_dict[news_index])


np.save('news_dict_test.npy', news_dict)                            # 保存新闻字符串字典
np.save('news_id_dict_test.npy', news_id_dict)                      # 保存新闻ID字典
new_corpus = np.array(new_corpus,dtype=object)                      # 保存新闻语料库
np.save('new_corpus_test.npy',new_corpus)
corpus = new_corpus


user_id_dict = {}
r_user_tsv = 'test/test.tsv'
w_user_tsv = 'test/test1.tsv'
'''
# 调试时为了节省时间可以调用已经保存好的模型
news_id_dict = np.load('news_id_dict_test.npy', allow_pickle=True).item()
news_dict = np.load('news_dict_test.npy', allow_pickle=True).item()
read_corpus = np.load('new_corpus_test.npy', allow_pickle=True)
corpus = read_corpus.tolist()
'''

# 将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
# 计算个词语出现的次数
X = vectorizer.fit_transform(corpus)
# 获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
# 类调用
transformer = TfidfTransformer()
# 将词频矩阵X统计成TF-IDF值
tfidf = transformer.fit_transform(X)

# 开始预测
news_item_map_dict = {}
for news_id in news_id_dict:
    news_item_tfidf = tfidf[news_id_dict[news_id]]
    news_coo = news_item_tfidf.tocoo()
    news_col = news_coo.col
    news_data = news_coo.data
    news_item_map = {}
    for i in range(len(news_col)):
        news_item_map[news_col[i]] = news_data[i]
    news_item_map_dict[news_id] = news_item_map

total_users = len(open(r_user_tsv, 'r',encoding='UTF-8').readlines())-1  # 待预测的总数
out_content = []
with open(r_user_tsv,'r',encoding='UTF-8') as tsv_in:
    tsv_reader = csv.reader(tsv_in, delimiter='\t')
    users_tsv_labels = tsv_reader.__next__()
    # 记录相似度的最大值和最小值并据此将其映射到[0,1]区间
    max_value = 0
    min_value = 1
    for user_index in track(range(total_users), description='用户点击率预测中...'):
        record = tsv_reader.__next__()
        user_id_dict[record[0]] = user_index
        out_content.append([record[0],record[1],record[2],record[3]])
        preq_dict = []
        user_history = [news_id_dict[str] for str in record[2].split(' ')]
        user_pref = get_user_pref(user_history)
        user_coo = user_pref.tocoo()
        user_col = user_coo.col
        user_data = user_coo.data
        user_pref_map = {}
        for i in range(len(user_col)):
            user_pref_map[user_col[i]] = user_data[i]
        news_test_list = [str for str in record[3].split(' ')]
        for news_item in news_test_list:
            news_item_map = news_item_map_dict[news_item]
            news_data = news_item_map.values()
            up = 0
            for key in user_pref_map.keys():
                if (key in news_item_map.keys()):
                    up += user_pref_map[key] * news_item_map[key]
            down = math.sqrt(sum([i * i for i in user_data])) * math.sqrt(sum([i * i for i in news_data]))
            max_value = max(max_value,up/down)
            min_value = min(min_value,up/down)
            preq_dict.append(up/down)
        out_content[user_index].append(preq_dict)

out_content = np.array(out_content,dtype=object)                                       # 保存第一次计算结果
np.save('out_content.npy',out_content)

read_out_content = np.load('out_content.npy', allow_pickle=True)
out_content = read_out_content.tolist()


# 将预测值写入表格
with open(w_user_tsv, 'w', newline='') as tsv_out:
    tsv_writer = csv.writer(tsv_out, delimiter='\t')
    tsv_writer.writerow(['Uid', 'Date', 'History', 'Impression', 'Predict'])
    for user_index in track(range(total_users), description='预测值写入中...'):
        row_content = out_content[user_index]
        maxi = len(row_content[4])
        for i in range(maxi):
            row_content[4][i] = float("{:.4f}".format((float(row_content[4][i])-min_value)/(max_value-min_value)))
        tsv_writer.writerow(row_content)

# 写文件
os.remove('test/test.tsv')
os.rename('test/test1.tsv', 'test/test.tsv')
print('预测完毕！')
######################################################################

