import os
print('执行中，请稍等...')
print('请输入数字选择模式，1代表调参模式，2代表预测模式：')
print('（备注：测试模式不写csv，按照8：2划分数据集，输出MAE和RMSE；预测模式将把预测结果写入test.csv）')
op1 = int(input())
while(op1 != 1 and op1 != 2):
    print('输入数字无效，请重新输入！')
    print('请输入数字选择核心算法：1代表UCF，2代表ICF：')
    op1 = int(input())
print('请输入数字选择核心算法：1代表UCF，2代表ICF：')
op2 = int(input())
while(op2 != 1 and op2 != 2):
    print('输入数字无效，请重新输入！')
    print('请输入数字选择核心算法：1代表UCF，2代表ICF：')
    op2 = int(input())
print('数据初始化中，请稍等...')
if(op1 == 1):
    if(op2 == 1):
        os.system('UCF_test.py')
    else:
        os.system('ICF_test.py')
else:
    if(op2 == 1):
        os.system('UCF.py')
    else:
        os.system('ICF.py')