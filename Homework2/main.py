import os
print('执行中，请稍等...')
print('请输入数字选择模式，1代表预测模式，2代表模型训练模式：')
print('（备注：预测模式把结果写入test/test.tsv；模型训练模式只生成模型文件）')
op1 = int(input())
while(op1 != 1 and op1 != 2):
    print('请输入数字选择模式，1代表预测模式，2代表模型训练模式：')
    print('（备注：预测模式把结果写入test/test.tsv；模型训练模式只生成模型文件）')
    op1 = int(input())
if(op1 == 1):
    os.system('pred.py')
else:
    print('请输入数字选择模型：1代表TF-IDF优化模型，2代表Doc2vec模型：')
    op2 = int(input())
    while (op2 != 1 and op2 != 2):
        print('请输入数字选择模型：1代表TF-IDF优化模型，2代表Doc2vec模型：')
        op2 = int(input())
    if(op2 == 1):
        os.system('news_preprocessing_tfidf.py')
    else:
        os.system('news_preprocessing_doc2vec.py')