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
        self.sim_userset = []

dataset = []
userset= []
business_avg_star = []

# 记录每个用户的信息
for i in range(user_index):
    user_tmp = User()
    userset.append(user_tmp)

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
# print("训练集大小={:d}, 测试集大小={:d}".format(len(trainset),len(testset)))

# 构建打分矩阵、时间戳矩阵
time_matrix = np.zeros([user_index, business_index])
scoring_matrix = np.zeros([user_index, business_index])
for item in trainset:
    time_matrix[item.user_id][item.business_id] = item.timeStamp
    scoring_matrix[item.user_id][item.business_id] = item.star
    userset[item.user_id].rated_business.append(item.business_id)

# 计算每个用户的平均打分
mean_rate_of_all_users = 0.0
for i in range(user_index):
    sum = 0.0
    for business in userset[i].rated_business:
        sum += scoring_matrix[i][business]
    # 注意由于按照8:2划分数据集和测试集，可能会出现冷启动问题
    if(len(userset[i].rated_business) != 0):
        userset[i].avg_star = sum / len(userset[i].rated_business)
    mean_rate_of_all_users += userset[i].avg_star
mean_rate_of_all_users /= user_index

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

# 构造用户相似度矩阵
sim_matrix = np.zeros([user_index, user_index])
alpha = 1.0 / 18 #希望相差18个月，影响降为一半

# 选择相似度计算方式
print('请输入数字选择相似度计算方式，1代表Pearson相似度，2代表余弦相似度：')
op2 = int(input())
while(op2 != 1 and op2 != 2):
    print('输入数字无效，请重新输入！')
    print('请输入数字选择相似度计算方式，1代表Pearson相似度，2代表余弦相似度：')
    op2 = int(input())

print('计算中，请稍等...')
for i in range(user_index-1):
    for j in range(i+1, user_index):
        # 计算Pearson相似性，将非负值纳入计算范围
        if(op2 == 1):
            P = list(set(userset[i].rated_business).intersection(set(userset[j].rated_business)))
            if(len(P) > 0):
                up = 0.0
                downa = 0.0
                downb = 0.0
                for business in P:
                    up += (scoring_matrix[i][business]-userset[i].avg_star)*(scoring_matrix[j][business]-userset[j].avg_star)
                    downa += (scoring_matrix[i][business]-userset[i].avg_star)**2
                    downb += (scoring_matrix[j][business]-userset[j].avg_star)**2
                downa = downa**0.5
                downb = downb**0.5
                if(downa*downb<1e-6):
                    continue
                sim_matrix[i][j] = up/(downa*downb)
                sim_matrix[j][i] = sim_matrix[i][j]
                if sim_matrix[i][j] > 0.5:
                    userset[i].sim_userset.append(j)
                    userset[j].sim_userset.append(i)
        # 计算余弦相似性
        else:
            Na = []
            Nb = []
            # 用户a和b给出积极评价的商品集合，这里以大于等于用户的历史平均打分为评价标准
            for business in userset[i].rated_business:
                if(scoring_matrix[i][business] >= userset[i].avg_star):
                    Na.append(business)
            for business in userset[j].rated_business:
                if(scoring_matrix[j][business] >= userset[j].avg_star):
                    Nb.append(business)
            N = list(set(Na).intersection(set(Nb)))
            if(len(N) > 0):
                up = 0
                down = (len(Na)*len(Nb))**0.5
                for business in N:
                    up += 1/(1+alpha*math.fabs(time_matrix[i][business]-time_matrix[j][business]))
                if(down > 1e-6):
                    sim_matrix[i][j] = up/down
                    sim_matrix[j][i] = sim_matrix[i][j]


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
    sim_userset = userset[user_id].sim_userset
    star = item.star
    up = 0.0
    down = 0.0

    '''
    处理冷启动问题：
    如果某用户没有任何行为数据，则预测值将取所有其他用户对该物品打分的平均值
    如果要预测的用户和物品都没有任何数据，取所有用户平均打分
    '''
    # 若该用户之前对该物品打过分，直接取原来的分值
    if scoring_matrix[user_id][business_id] > 1e-6:
        preq = scoring_matrix[user_id][business_id]
    elif (userset[user_id].avg_star > 1e-6):
        up = 0.0
        down = 0.0
        preq = userset[user_id].avg_star
        for sim_user in sim_userset:
            if (scoring_matrix[sim_user][business_id] > 1e-6):
                up += sim_matrix[user_id][sim_user] * (
                            scoring_matrix[sim_user][business_id] - userset[sim_user].avg_star) * (
                                  1 / (1 + alpha * math.fabs(timeStamp - time_matrix[sim_user][business_id])))
                down += sim_matrix[user_id][sim_user]
        # python精度问题，防止除零
        if (down > 1e-6):
            preq += up / down
    elif business_avg_star[business_id] > 1e-6:
        preq = business_avg_star[business_id]
    else:
        preq = mean_rate_of_all_users

    sum_MAE += math.fabs(preq-star)
    sum_RMSE += (preq-star)**2

MAE = sum_MAE/size
RMSE = (sum_RMSE/size)**0.5

print("MAE = {:.3f}".format(MAE))
print("RMSE = {:.3f}".format(RMSE))
