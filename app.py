# -*- coding:utf-8 -*-
from flask import Flask, render_template
import pandas as pd
import json
import networkx as nx

def load_wb_data(path):
    weibo = pd.read_csv(path)
    weibo['user'] = weibo['user'].map(eval)    #将读取到的str数据转为dict类型
    num_wb = len(weibo)
    return weibo, num_wb



app = Flask(__name__)
app.secret_key = 'lisenzzz'
weibo, num_wb = load_wb_data('static/data/weibo/weibo.csv')


@app.route('/')
def hello_world():
    return render_template('index.html')


# 选择单个影响力最大的种子基于ic模型（每个节点模拟一次）
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



if __name__ == '__main__':
    app.run(debug=True)
