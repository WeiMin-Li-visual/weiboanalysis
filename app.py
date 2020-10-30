# -*- coding:utf-8 -*-
from flask import Flask, render_template,request
import pandas as pd
import json
import snownlp as sn
from snownlp import sentiment
import jieba
import re

def load_wb_data(path):
    weibo = pd.read_csv(path)
    weibo['user'] = weibo['user'].map(eval)    #将读取到的str数据转为dict类型
    num_wb = len(weibo)
    return weibo, num_wb

#分词，去除停用词、英文、符号和数字等
def clearTxt(sentence):
    if sentence != '':
        sentence = sentence.strip()  # 去除文本前后空格
        # 去除文本中的英文和数字
        sentence = re.sub("[a-zA-Z0-9]", "", sentence)
        # 去除文本中的中文符号和英文符号
        sentence = re.sub("[\s+\.\!\/_,$%^*(+]\"\']+|[+——！，:\[\]。!：？?～”,、\.\/~@#￥%……&*【】 （）]+", "", sentence)
        sentence =jieba.lcut(sentence,cut_all=False)
        stopwords = [line.strip() for line in
                     open('static/data/weibo/analysis_cache/stopwords.txt', encoding='gbk').readlines()]
        outstr = ''
        # 去停用词
        for word in sentence:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        # print(outstr)
        return outstr

def savemodel_snownlp():
    weibo_80k_data = pd.read_csv('static/data/weibo/weibo_senti_80k.csv', encoding='utf-8')
    col_label = weibo_80k_data.iloc[:, 0].values
    col_content = weibo_80k_data.iloc[:, 1].values
    weibodata = []
    for i in range(len(col_label)):
        weibodata.append([col_label[i], clearTxt(col_content[i])])
    weibodata = pd.DataFrame(weibodata)
    weibodata.columns = ['label', 'comment']
    x = weibodata[['comment']]
    y = weibodata.label
    x = x.comment.apply(clearTxt)
    neg_file = open('static/data/weibo/analysis_cache/neg_file.txt', 'w+', encoding='utf-8')
    pos_file = open('static/data/weibo/analysis_cache/pos_file.txt', 'w+', encoding='utf-8')
    for i in range(len(weibo_80k_data)):
        if y[i] == 0:
            neg_file.write(clearTxt(x[i]) + '\n')
        else:
            pos_file.write(clearTxt(x[i]) + '\n')
    sentiment.train('static/data/weibo/analysis_cache/neg_file.txt',
                    'static/data/weibo/analysis_cache/pos_file.txt')  # 训练语料库
    # 保存模型
    sentiment.save('static/data/weibo/analysis_cache/sentiment_snownlp.marshal')

app = Flask(__name__)
app.secret_key = 'lisenzzz'
weibo, num_wb = load_wb_data('static/data/weibo/weibo.csv')


@app.route('/')
def hello_world():
    return render_template('index.html')

# 微博数据来源可视化

@app.route('/wbstatistic')
def wb_statistic():
    data = {}    #所有要传给前端的数据

    # 统计微博的来源分布
    source_count = weibo.loc[:, 'source'].value_counts()  # 微博来源数量统计
    num_src = 10                                          # 要展示的来源数量
    other_count = source_count[num_src:-1].sum()          # 其它来源的总数量
    src_dic = {'其它来源':int(other_count)}               # 来源分布，dict类型
    for i in range(num_src):
        index = num_src-i-1
        source = source_count.index[index]
        count = source_count[index]
        src_dic.update({source: int(count)})
    src_rate = {'其它来源': int(other_count)/num_wb}       # 来源比率，dict类型
    for i in range(num_src):
        index = num_src - i - 1
        source = source_count.index[index]
        count = source_count[index]
        src_rate.update({source: int(count)/num_wb})
    #传给前端的微博来源分布数据
    source = {
        "num_wb": int(num_wb),
        "src_count": src_dic,
        "src_rate": src_rate
    }
    data['source'] = source


    data_json = json.dumps(data)
    return render_template('wbstatistic.html', data_json=data_json)

#单条语句情感极性测试，返回极性值
@app.route('/single_sentiment',methods=['GET','POST'])
def senti_test():
    global senti_value
    global content_err
    if request.method =='POST':
        content=request.form.get('content_value')
        if content!='':
            content_err=1
            # print(content)
            senti=sn.SnowNLP(clearTxt(content))
            senti_value=round((senti.sentiments),2)
        else:
            senti_value=0
            content_err=0
        senti_value = json.dumps(senti_value)
        content_err = json.dumps(content_err)
        return render_template('single_sentiment.html',senti_value=senti_value,content_err=content_err)
    else:
        # content_err = 0
        # senti_value = 0
        # senti_value = json.dumps(senti_value)
        # content_err = json.dumps(content_err)
        # return render_template('single_sentiment.html', senti_value=senti_value, content_err=content_err)
        return render_template('single_sentiment.html')
if __name__ == '__main__':
    app.run(debug=True)
