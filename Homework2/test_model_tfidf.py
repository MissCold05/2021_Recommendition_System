######################################################################
# Author : 1018512015-张翼鹏
# Date : 2021/04/30

######################################################################
# 调库区
import csv
import math
import os
import numpy as np
from rich.progress import track
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
######################################################################

######################################################################
# 函数区
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
user_id_dict = {}
r_user_tsv = 'train/train.tsv'
w_user_tsv = 'Solution1.tsv'
news_id_dict = np.load('news_id_dict.npy', allow_pickle=True).item()
news_dict = np.load('news_dict.npy', allow_pickle=True).item()
read_corpus = np.load('new_corpus.npy')
corpus = read_corpus.tolist()

#将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
#计算个词语出现的次数
X = vectorizer.fit_transform(corpus)
#获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
#类调用
transformer = TfidfTransformer()
#将词频矩阵X统计成TF-IDF值
tfidf = transformer.fit_transform(X)

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

total_users = len(open(r_user_tsv, 'r',encoding='UTF-8').readlines())-1  # 用户总数
if not os.path.exists('out_content.npy'):
    out_content = []
    with open(r_user_tsv,'r',encoding='UTF-8') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')
        users_tsv_labels = tsv_reader.__next__()
        # 记录相似度的最大值和最小值并据此将其映射到[0,1]区间
        max_value = 0
        min_value = 1
        x_record = []
        y_record = []
        for user_index in track(range(total_users), description='用户信息处理中...'):
            record = tsv_reader.__next__()
            user_id_dict[record[0]] = user_index
            out_content.append([record[0],record[3]])
            preq_dict = []
            user_history = [news_id_dict[str] for str in record[2].split(' ')]
            user_pref = get_user_pref(user_history)
            user_coo = user_pref.tocoo()
            user_col = user_coo.col
            user_data = user_coo.data
            user_pref_map = {}
            for i in range(len(user_col)):
                user_pref_map[user_col[i]] = user_data[i]
            news_test_list = [str[:-2] for str in record[3].split(' ')]
            for news_item in news_test_list:
                news_item_map = news_item_map_dict[news_item]
                news_data = news_item_map.values()
                up = 0
                for key in user_pref_map.keys():
                    if (key in news_item_map.keys()):
                        up += user_pref_map[key] * news_item_map[key]
                down = math.sqrt(sum([i * i for i in user_data])) * math.sqrt(sum([i * i for i in news_data]))
                max_value = max(max_value,up/down)
                x_record.append(up/down)
                y_record.append(0)
                min_value = min(min_value,up/down)
                preq_dict.append(up/down)
            out_content[user_index].append(preq_dict)

    out_content = np.array(out_content)                                       # 保存第一次计算结果
    np.save('out_content.npy',out_content)

read_out_content = np.load('out_content.npy', allow_pickle=True)
out_content = read_out_content.tolist()

max_value = 1.0
min_value = 0.0
# 若用该方法处理的新闻字典已经存在则不再重复操作
if not os.path.exists("Solution1.tsv"):
    with open(w_user_tsv, 'w', newline='') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')
        tsv_writer.writerow(['Uid', 'Actual_values', 'Pred_values', 'Avg_RMD'])
        RMD_SUM = 0
        print('开始')
        for user_index in track(range(total_users), description='用户点击率预测中...'):
            row_content = out_content[user_index]
            row_content[1] = row_content[1].split(' ')
            RMD_sum = 0
            maxi = len(row_content[2])
            for i in range(maxi):
                row_content[2][i] = float("{:.4f}".format((float(row_content[2][i])-min_value)/(max_value-min_value)))
                RMD_sum += abs(float(row_content[2][i]) - float(row_content[1][i][-1]))
            RMD_sum /= maxi
            RMD_SUM += RMD_sum
            row_content.append(str(RMD_sum))
            tsv_writer.writerow(row_content)
        RMD_SUM /= total_users
        print('结束')
        print('平均预测误差为：{:.4f}'.format(RMD_SUM))



