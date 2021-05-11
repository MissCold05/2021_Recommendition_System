######################################################################
# Author : 1018512015-张翼鹏
# Date : 2021/04/29

######################################################################
# 调库区
import os
import csv
from rich.progress import track
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

######################################################################

stop_words = set(stopwords.words('english'))
if not os.path.exists('d2v.model'):
    news_tags = []
    news_txt_data =[]
    r_news_tsv = 'train/train_news.tsv'
    total_news = len(open(r_news_tsv, 'r',encoding='UTF-8').readlines())-1  # 新闻总数
    with open(r_news_tsv,'r',encoding='UTF-8') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')
        news_tsv_labels = tsv_reader.__next__()
        for news_index in track(range(total_news),description='新闻文本读取中...'):
            record = tsv_reader.__next__()
            news_tags.append(record[0])
            news_txt_data.append(record[3]+'. '+record[4])

    tagged_data = [TaggedDocument(words=[word for word in word_tokenize(_d.lower()) if word not in stop_words], tags=news_tags)
                   for k, _d in enumerate(news_txt_data)]

    # 模型训练参数
    max_epochs = 100
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)

    model.build_vocab(tagged_data)

    for epoch in track(range(max_epochs),description='模型训练中...'):
        print('第{0}次迭代'.format(epoch))
        model.train(tagged_data, epochs=model.epochs, total_examples=model.corpus_count)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    model.save("d2v.model")
    print('模型训练完毕！')