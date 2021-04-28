'''
测试文件，不写csv，仅输出MAE和RMSE供参考
'''
# 导入所需的包
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# 读取train.csv文件
f = open('train.csv')
df = pd.read_csv(f)
f.close()

# 构造用户和商品的id字典
user_index = 0
business_index = 0
user_dict = {}
business_dict = {}
for index, row in df.iterrows():
    user_id = row["user_id"]
    business_id = row["business_id"]
    if user_id not in user_dict:
        user_dict[user_id] = user_index
        user_index += 1
    if business_id not in business_dict:
        business_dict[business_id] = business_index
        business_index += 1

# 数据类，每条数据包括哪位用户什么时刻给哪个物品打了几分
class Data(object):
    # 初始化给对象属性赋值
    def __init__(self, user_id, business_id, timeStamp, star):
        self.user_id = user_id
        self.business_id = business_id
        self.timeStamp = timeStamp
        self.star = star

# 用户类，每个用户记录其打过分的物品、历史平均打分和与之相似的用户集合
class User(object):
    def __init__(self):
        self.rated_business = []
        self.avg_star = 0.0

# 商品类，记录对该商品给出过积极评价的用户集合
class Business(object):
    def __init__(self):
        self.sim_business = []
        self.user_with_positive_feedback = []

dataset = []
userset = []
business_set = []
business_avg_star = []

# 记录每个用户的信息
for i in range(user_index):
    user_tmp = User()
    userset.append(user_tmp)

# 记录每个商品的信息
for i in range(business_index):
    business_tmp = Business()
    business_set.append(business_tmp)

# 记录每个物品被打的平均分值
for i in range(business_index):
    business_avg_star.append(0.0)

# 初始化数据
for index, row in df.iterrows():
    user_id = user_dict[row["user_id"]]
    business_id = business_dict[row["business_id"]]
    timeStamp = time.mktime(datetime.strptime(row["date"], "%Y/%m/%d %H:%M").timetuple())/(30*24*3600)
    data = Data(user_id, business_id, timeStamp, int(row["stars"]))
    dataset.append(data)

# 按照8:2的比例划分训练集和测试集
trainset, testset = train_test_split(dataset, test_size=0.2)

# 构建打分矩阵、时间戳矩阵
time_matrix = np.zeros([user_index, business_index])
scoring_matrix = np.zeros([user_index, business_index])
for item in trainset:
    time_matrix[item.user_id][item.business_id] = item.timeStamp
    scoring_matrix[item.user_id][item.business_id] = item.star
    userset[item.user_id].rated_business.append(item.business_id)

# 计算每个用户的平均打分
mean_rate_of_all_users = 0.0 # 打分表中所有分值的平均值
for i in range(user_index):
    sum = 0.0
    for business in userset[i].rated_business:
        sum += scoring_matrix[i][business]
    # 注意由于按照8:2划分数据集和测试集，可能会出现冷启动问题
    if(len(userset[i].rated_business) != 0):
        userset[i].avg_star = sum / len(userset[i].rated_business)
    mean_rate_of_all_users += userset[i].avg_star
mean_rate_of_all_users /= user_index

# 记录给每个物品积极评价的用户集合
for i in range(business_index):
    for j in range(user_index):
        # 用户给该物品的打分大于等于该用户历史平均打分
        if(scoring_matrix[j][i] >= userset[j].avg_star):
            business_set[i].user_with_positive_feedback.append(j)

# 构造物品被打分的向量
for i in range(business_index):
    sum = 0.0
    cnt = 0
    for j in range(user_index):
        if scoring_matrix[j][i] < 1e-6:
            continue
        cnt += 1
        sum += scoring_matrix[j][i]
    if(cnt != 0):
        business_avg_star[i] = sum/cnt

# 构造商品相似度矩阵
sim_matrix = np.zeros([business_index, business_index])
alpha = 1.0 / 36 # 希望相差三年，影响降为一半
beta = 1.0 / 36 # 希望相差三年，影响降为一半

# 选择相似度计算方式
print('请输入数字选择相似度计算方式，1代表改进的余弦相似度，2代表引入时间因素的余弦相似度：')
op2 = int(input())
while(op2 != 1 and op2 != 2):
    print('输入数字无效，请重新输入！')
    print('请输入数字选择相似度计算方式，1代表改进的余弦相似度，2代表引入时间因素的余弦相似度：')
    op2 = int(input())

print('计算中，请稍等...')
for i in range(business_index-1):
    for j in range(i+1, business_index):
        # 计算改进的余弦相似性，将非负值纳入相似范围
        Na = business_set[i].user_with_positive_feedback
        Nb = business_set[j].user_with_positive_feedback
        N = list(set(Na).intersection(set(Nb)))
        up = 0.0
        down = (len(Na) * len(Nb)) ** 0.5
        if(op2 == 1):
            for user in N:
                Nu = userset[user].rated_business
                if(len(Nu) > 1e-6):
                    up += 1/(math.log(1+len(Nu)))
        # 计算引入时间的余弦相似性，将非负值纳入相似范围
        else:
            for user in N:
                up += 1/(1+alpha*math.fabs(time_matrix[user][i]-time_matrix[user][j]))
        if (down > 1e-6):
            sim_matrix[i][j] = up / down
            sim_matrix[j][i] = sim_matrix[i][j]
        # 值非负认为相似
        if(sim_matrix[i][j] > 1e-6):
            business_set[i].sim_business.append(j)
            business_set[i].sim_business.append(j)


# 评估预测结果
sum_MAE = 0.0
sum_RMSE = 0.0
mean_bias = 0.0 #平均误差
valid_num = 0
size = len(testset)
for item in testset:
    user_id = item.user_id
    business_id = item.business_id
    timeStamp = item.timeStamp
    star = item.star

    '''
    处理冷启动问题：
    如果某用户没有任何行为数据，则预测值将取所有其他用户对该物品打分的平均值
    如果要预测的用户和物品都没有任何数据，取所有用户平均打分
    '''
    # 若该用户之前对该物品打过分，直接取原来的分值
    if scoring_matrix[user_id][business_id] > 1e-6:
        preq = scoring_matrix[user_id][business_id]
    else:
        preq = business_avg_star[business_id]
        rated_item = []
        for sim_business in userset[user_id].rated_business:
            if(sim_matrix[business_id][sim_business] > 1e-6):
                rated_item.append(sim_business)
        up = 0.0
        down = 0.0
        for sim_business in rated_item:
            up += sim_matrix[business_id][sim_business]*scoring_matrix[user_id][sim_business]*(1/(1+beta*math.fabs(timeStamp-time_matrix[user_id][sim_business])))
            down += sim_matrix[business_id][sim_business]
        if(down > 1e-6):
            preq = up/down

    sum_MAE += math.fabs(preq-star)
    sum_RMSE += (preq-star)**2

MAE = sum_MAE/size
RMSE = (sum_RMSE/size)**0.5

print("MAE = {:.3f}".format(MAE))
print("RMSE = {:.3f}".format(RMSE))
