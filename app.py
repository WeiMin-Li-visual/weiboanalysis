# -*- coding:utf-8 -*-
from flask import Flask, render_template, request
import pandas as pd
import json
import snownlp as sn
from snownlp import sentiment
import jieba
import wordcloud
import numpy as np
import re
import matplotlib.pyplot as plt
import glob
import imageio
import os
from wordcloud import WordCloud,ImageColorGenerator
from snownlp import sentiment
from PIL import Image,ImageDraw,ImageFont
from os import path

data_path='static/data/weibo'
data_path_cache='static/data/weibo/analysis_cache'

def load_wb_data(path):
    weibo = pd.read_csv(path)
    weibo['user'] = weibo['user'].map(eval)  # 将读取到的str数据转为dict类型
    num_wb = len(weibo)
    return weibo, num_wb

# 分词，去除停用词、英文、符号和数字等
def clearTxt(sentence):
    if sentence != '':
        sentence = sentence.strip()  # 去除文本前后空格
        # 去除文本中的英文和数字
        sentence = re.sub("[a-zA-Z0-9]", "", sentence)
        # 去除文本中的中文符号和英文符号
        sentence = re.sub("[\s+\.\!\/_,$%^*(+]\"\']+|[+——！，:\[\]。!：？?～”,、\.\/~@#￥%……&*【】 （）]+", "", sentence)
        sentence = jieba.lcut(sentence, cut_all=False)
        stopwords = [line.strip() for line in
                     open(data_path_cache+'/stopwords.txt', encoding='gbk').readlines()]
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
    weibo_80k_data = pd.read_csv(data_path+'/weibo_senti_80k.csv', encoding='utf-8')
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
    neg_file = open(data_path_cache+'/neg_file.txt', 'w+', encoding='utf-8')
    pos_file = open(data_path_cache+'/pos_file.txt', 'w+', encoding='utf-8')
    for i in range(len(weibo_80k_data)):
        if y[i] == 0:
            neg_file.write(clearTxt(x[i]) + '\n')
        else:
            pos_file.write(clearTxt(x[i]) + '\n')
    sentiment.train(data_path_cache+'/neg_file.txt',
                    data_path_cache+'/pos_file.txt')  # 训练语料库
    # 保存模型
    sentiment.save(data_path_cache+'/sentiment_snownlp.marshal')

#求文本的情感倾向值，>0.57则默认为积极，<0.5则默认为消极，0.57与0.5之间可默认为中性
def sen_value(text):
    senti = sn.SnowNLP(text)
    senti_value = round((senti.sentiments), 2)
    return senti_value

#统计所有文本情感极性，并生成词云图，不考虑时间、省份
def senti_diffusion():
    data = pd.read_csv(data_path + '/weibo.csv', encoding='utf-8')
    content = data.iloc[:, 2].values
    count=[0,0,0]#统计极性，积极，中性，消极
    for i in range(len(data)):
        data.iloc[i,2]=clearTxt(str(data.iloc[i,2]))#处理文本
        if data.iloc[i, 2] == '':
            data.iloc[i, 2] = '年月日'  # 即默认为中性态度
        senti_value = sen_value(data.iloc[i,2])
        if (senti_value >= 0.57):
            count[0] += 1
        elif (senti_value <= 0.5):
            count[2] += 1
        elif (senti_value > 0.5 and senti_value < 0.57):
            count[1] += 1
    print(count)
    # 统计结果：[114092, 10052, 65256]
    #生成词云
    text=''
    for j in range(len(data)):
        line = re.sub("[\s+\.\!\/_,$%^*(+]\"\']+|[+——！，:\[\]。!：？?～”,、\.\/~@#￥%……&*【】 （）]+", "",
                      str(data.iloc[j, 2]))  # 去除符号
        text += ' '.join(jieba.cut(line, cut_all=True))
        # 设置词云格式
    backgroud_image = np.array(Image.open(r"static/image/wc_time.jpg"))
    wc = WordCloud(
        background_color='white',  # 设置背景颜色，与图片的背景色相关
        mask=backgroud_image,  # 设置背景图片
        font_path='C:\Windows\Fonts\STZHONGS.TTF',  # 显示中文，可以更换字体
        max_words=300,  # 设置最大显示的字数
        stopwords={'网页', '链接','博转发'},  # 设置停用词，停用词则不再词云图中表示
        max_font_size=80,  # 设置字体最大值
        random_state=5,  # 设置有多少种随机生成状态，即有多少种配色方案
        scale=1,  # 设置生成的词云图的大小
    ).generate(text)  # 生成词云
    image_colors = ImageColorGenerator(backgroud_image)
    plt.imshow(wc.recolor(color_func=image_colors))  # 显示词云图
    plt.axis('off')  # 不显示x,y轴坐

    wc.to_file(path.join(data_path_cache + '/weibo_ciyun.png'))
    return count


#根据时间信息判断所有用户的情感倾向，生成词云图
def senti_diffusion_time():
    time_data=pd.read_excel(data_path_cache+'/weibo_time.xlsx',encoding='utf-8',sheet_name=None)
    sheet_name=time_data.keys()#获取当前表格所有sheet名字
    year_count=[]#保存每一年的积极、消极以及中性文本数目
    for i in sheet_name:
        time_data_y=time_data[i]#当前某一年的所有用户数据
        positive=0
        negative=0
        neutral=0
        wc_text=''#当前年份的词云文本

        #根据年份统计文本情感倾向
        for j in range(len(time_data_y)):
            time_data_y.iloc[j, 2]=clearTxt(str(time_data_y.iloc[j,2]))#由于部分数据只有数字或者符号，需要转成字符型
            if time_data_y.iloc[j,2]=='':
                time_data_y.iloc[j, 2]='年月日'#即默认为中性态度
            senti_value=sen_value(time_data_y.iloc[j,2])#time_data_y.iloc[j,2]表示表格中第三列文本数据
            if(senti_value>=0.57):
                positive+=1
            elif(senti_value<=0.5):
                negative+=1
            elif(senti_value>0.5 and senti_value<0.57):
                neutral+=1
        year_count.append([positive, neutral, negative])  # 保存每一年的积极、消极以及中性文本数目
        # [[3, 5, 13], [763, 105, 576], [876, 109, 805], [1453, 140, 1095], [5627, 497, 3144], [2916, 420, 1767],
        #  [3398, 502, 3131], [7328, 940, 6149], [15219, 1579, 10897], [16276, 1868, 13630], [26454, 3719, 23122]]

        #根据年份文本形成词云图
        for j in range(len(time_data_y)):
            line=re.sub("[\s+\.\!\/_,$%^*(+]\"\']+|[+——！，:\[\]。!：？?～”,、\.\/~@#￥%……&*【】 （）]+", "", str(time_data_y.iloc[j, 2]))#去除符号
            wc_text += ' '.join(jieba.cut(line,cut_all=True))
        #设置词云格式
        backgroud_image = np.array(Image.open(r"static/image/wc_time.jpg"))
        wc_year = WordCloud(
            background_color='white',  # 设置背景颜色，与图片的背景色相关
            mask=backgroud_image,  # 设置背景图片
            font_path='C:\Windows\Fonts\STZHONGS.TTF',  # 显示中文，可以更换字体
            max_words=300,  # 设置最大显示的字数
            stopwords={'网页','链接'},  # 设置停用词，停用词则不再词云图中表示
            max_font_size=80,  # 设置字体最大值
            random_state=5,  # 设置有多少种随机生成状态，即有多少种配色方案
            scale=1,  # 设置生成的词云图的大小
        ).generate(wc_text)#生成词云
        image_colors = ImageColorGenerator(backgroud_image)
        plt.imshow(wc_year.recolor(color_func=image_colors))  # 显示词云图
        # plt.imshow(wc_year, interpolation='bilinear')
        plt.axis('off')#不显示x,y轴坐
        # 按递增顺序保存生成的词云图

        wc_year.to_file(path.join(data_path_cache+'/time_wc', str(i) + '_wc.png'))

    # 生成动态gif文件
    def create_gif(image_list, gif_name):
        frames = []
        for image_name in image_list:
            frames.append(imageio.imread(image_name))
        # 保存gif文件
        imageio.mimsave(gif_name, frames, 'gif', duration=0.3)
        return
    def find_all_gif():

        png_filenames = glob.glob(data_path_cache+'/result/time_wc_chuo/*')  # 加入图片位置，绝对路径
        buf = []
        for png_file in png_filenames:
            buf.append(png_file)
        return buf
    #为每张图片生成时间戳
    png_list=os.listdir(data_path_cache+'/time_wc')
    font=ImageFont.truetype("C:\Windows\Fonts\STZHONGS.TTF",40)
    j=0
    dir_list = os.listdir(data_path_cache+'/time_wc/')
    dir = []

    for i in dir_list:
        dir.append(re.sub('_wc','',str(os.path.splitext(i)[0])))
    for png in png_list:
        name = png
        imageFile = data_path_cache+'/time_wc/' + name
        im = Image.open(imageFile)
        draw = ImageDraw.Draw(im)
        draw.text((0, 0),  # 设置字体位置
                  dir[j]+'年',  # 设置内容
                  (0, 0, 0),  # 设置颜色
                  font=font)  # "设置字体
        draw = ImageDraw.Draw(im)
        # 存储图片
        im.save(data_path_cache+'/result/time_wc_chuo/'+ name)
        j += 1
    buff=find_all_gif()
    time_wc=create_gif(buff,data_path_cache+'/result/time_wc.gif') #生成时间词云动态gif图
    return year_count,time_wc

#根据地域信息判断所有用户的情感倾向，生成词云图
def senti_diffusion_position():
    position_data=pd.read_excel(data_path_cache+'/weibo_position.xlsx',encoding='utf-8',sheet_name=None)
    sheet_name=position_data.keys()#获取当前表格所有sheet名字
    # print(sheet_name)
    position_count=[]#保存每个省份的积极、消极以及中性文本数目
    for i in sheet_name:
        position_data_y=position_data[i]#当前某省份的所有用户数据
        positive = 0
        negative = 0
        neutral = 0
        wc_text = ''  # 当前省份的词云文本

        # 根据省份统计文本情感倾向
        for j in range(len(position_data_y)):
            position_data_y.iloc[j, 2] = clearTxt(str(position_data_y.iloc[j, 2]))  # 由于部分数据只有数字或者符号，需要转成字符型
            if position_data_y.iloc[j, 2] == '':
                position_data_y.iloc[j, 2] = '年月日'  # 即默认为中性态度
            senti_value = sen_value(position_data_y.iloc[j, 2])  # position_data_y.iloc[j,2]表示表格中第三列文本数据
            if (senti_value >= 0.57):
                positive += 1
            elif (senti_value <= 0.5):
                negative += 1
            elif (senti_value > 0.5 and senti_value < 0.57):
                neutral += 1
        position_count.append([i,positive, neutral, negative])  # 保存每个省份的积极、消极以及中性文本数目
# [['黑龙', 1428, 246, 1270], ['北京', 5505, 677, 4699], ['辽宁', 8121, 1290, 6342], ['内蒙', 1149, 165, 1121],
#  ['香港', 577, 67, 431], ['天津', 1107, 138, 922], ['云南', 1240, 129, 1007], ['湖南', 2125, 194, 1533],
#  ['河南', 2083, 230, 1621], ['山东', 2873, 250, 2488], ['西藏', 370, 51, 438], ['广西', 1077, 131, 962],
#  ['山西', 1356, 127, 1359], ['台湾', 818, 56, 444], ['新疆', 781, 86, 666], ['江西', 983, 140, 1218], ['吉林', 2066, 273, 1208],
#  ['河北', 1928, 225, 1507], ['四川', 2518, 299, 2027], ['甘肃', 1001, 131, 695], ['福建', 2465, 344, 1761],
#  ['广东', 6847, 912, 5948], ['安徽', 1425, 132, 962], ['浙江', 3947, 406, 3044], ['上海', 3677, 384, 2987],
#  ['陕西', 1744, 160, 1163], ['澳门', 474, 51, 484], ['海外', 4319, 628, 2467], ['江苏', 4027, 464, 3104],
#  ['湖北', 1712, 193, 1362], ['海南', 697, 136, 663], ['贵州', 1025, 106, 806], ['重庆', 1275, 167, 987],
#  ['其他', 8699, 966, 6681], ['青海', 637, 57, 476], ['宁夏', 397, 24, 275]]

        # 根据省份文本形成词云图
        for j in range(len(position_data_y)):
            line = re.sub("[\s+\.\!\/_,$%^*(+]\"\']+|[+——！，:\[\]。!：？?～”,、\.\/~@#￥%……&*【】 （）]+", "",
                          str(position_data_y.iloc[j, 2]))  # 去除符号
            wc_text += ' '.join(jieba.cut(line, cut_all=True))
        # 设置词云格式
        backgroud_Image = np.array(Image.open("static/image/wc_time.jpg"))
        wc_position = WordCloud(
            background_color='white',  # 设置背景颜色，与图片的背景色相关
            mask=backgroud_Image,  # 设置背景图片
            font_path='C:\Windows\Fonts\STZHONGS.TTF',  # 显示中文，可以更换字体
            max_words=300,  # 设置最大显示的字数
            stopwords={'网页', '链接'},  # 设置停用词，停用词则不再词云图中表示
            max_font_size=80,  # 设置字体最大值
            random_state=5,  # 设置有多少种随机生成状态，即有多少种配色方案
            scale=1 # 设置生成的词云图的大小
        )
        wc_position.generate(wc_text)  # 生成词云
        image_colors = ImageColorGenerator(backgroud_Image)
        plt.imshow(wc_position.recolor(color_func=image_colors))  # 显示词云图
        plt.axis('off')  # 不显示x,y轴坐标
        # 按递增顺序保存生成的词云图
        wc_position.to_file(path.join(data_path_cache + '/position_wc', str(i) + '_wc.png'))

    # 生成动态gif文件
    def create_gif(image_list, gif_name):
        frames = []
        for image_name in image_list:
            frames.append(imageio.imread(image_name))
        # 保存gif文件
        imageio.mimsave(gif_name, frames, 'gif', duration=0.3)
        return

    def find_all_gif():
        png_filenames = glob.glob(data_path_cache + '/result/position_wc_chuo/*')  # 加入图片位置，绝对路径
        buf = []
        for png_file in png_filenames:
            buf.append(png_file)
        return buf

    # 为每张图片生成地域省份戳
    png_list = os.listdir(data_path_cache + '/position_wc')
    font = ImageFont.truetype("C:\Windows\Fonts\STZHONGS.TTF", 40)
    j = 0
    dir_list = os.listdir(data_path_cache + '/position_wc/')
    dir = []
    for i in dir_list:
        position_n=os.path.splitext(i)[0]
        if position_n =='黑龙_wc':
            position_n='黑龙江_wc'
        if position_n =='内蒙_wc':
            position_n='内蒙古_wc'
        dir.append(re.sub('_wc', '', str(position_n)))
    for png in png_list:
        name = png
        imageFile = data_path_cache + '/position_wc/' + name
        im = Image.open(imageFile)
        draw = ImageDraw.Draw(im)
        draw.text((0, 0),  # 设置字体位置
                    dir[j] ,  # 设置内容
                    (0, 0, 0),  # 设置颜色
                    font=font)  # "设置字体
        draw = ImageDraw.Draw(im)
        # 存储图片
        im.save(data_path_cache + '/result/position_wc_chuo/' + name)
        j += 1
    buff = find_all_gif()
    position_wc = create_gif(buff, data_path_cache + '/result/position_wc.gif')  # 生成时间词云动态gif图
    return position_count, position_wc


app = Flask(__name__)
app.secret_key = 'lisenzzz'
weibo, num_wb = load_wb_data('static/data/weibo/weibo.csv')


@app.route('/')
def hello_world():
    return render_template('index.html')

# 微博数据来源可视化

@app.route('/wbstatistic')
def wb_statistic():
    data = {}  # 所有要传给前端的数据

    # 统计微博的来源分布
    source_count = weibo.loc[:, 'source'].value_counts()  # 微博来源数量统计
    num_src = 10  # 要展示的来源数量
    other_count = source_count[num_src:-1].sum()  # 其它来源的总数量
    src_dic = {'其它来源': int(other_count)}  # 来源分布，dict类型
    for i in range(num_src):
        index = num_src - i - 1
        source = source_count.index[index]
        count = source_count[index]
        src_dic.update({source: int(count)})
    src_rate = {'其它来源': int(other_count) / num_wb}  # 来源比率，dict类型
    for i in range(num_src):
        index = num_src - i - 1
        source = source_count.index[index]
        count = source_count[index]
        src_rate.update({source: int(count) / num_wb})
    # 传给前端的微博来源分布数据
    source = {
        "num_wb": int(num_wb),
        "src_count": src_dic,
        "src_rate": src_rate
    }
    data['source'] = source

    data_json = json.dumps(data)
    return render_template('wbstatistic.html', data_json=data_json)

# 单条语句情感极性测试，返回极性值
@app.route('/single_sentiment', methods=['GET', 'POST'])
def senti_test():
    global senti_value
    global content_err
    if request.method == 'POST':
        content = request.form.get('content_value')
        if content != '':
            content_err = 1
            senti_value = sen_value(clearTxt(content))
        else:
            senti_value = 0
            content_err = 0
        senti_value = json.dumps(senti_value)
        content_err = json.dumps(content_err)
        return render_template('single_sentiment.html', senti_value=senti_value, content_err=content_err)
    else:
        return render_template('single_sentiment.html')

# 所有数据情感分析
@app.route('/weibo_sentiment_analysis', methods=['GET', 'POST'])
def sentiment_analysis():
    with open(data_path_cache+"/result/user_position_count.txt", "r",
              encoding='utf-8') as user_position_count:
        data = user_position_count.readlines()
        user_position = []  # 用户省份地址
        user_position_c = []  # 省份用户人数
        user_text_count=[114092, 10052, 65256] #用户微博极性统计，根据senti_diffusion()获得
        for i in range(len(data)):
            user_position.append(data[i][0:2])
            user_position_c.append(int(data[i][3:]))

    def getresult(path):
        user_senticount=pd.read_table(path)
        return user_senticount

    if request.method == 'GET':
        user_position = json.dumps(user_position)
        user_position_c = json.dumps(user_position_c)
        user_text_count=json.dumps(user_text_count)
        user_ciyun_path=data_path_cache+'/weibo_ciyun.png'
        return render_template('weibo_sentiment_analysis.html', user_position_c=user_position_c,
                               user_position=user_position,user_text_count=user_text_count,user_ciyun_path=user_ciyun_path)

#根据时间以及省份给出极性分布及词云图
@app.route('/sentiment_detail', methods=['GET', 'POST'])
def sentiment_analysis_detail():
    # 按照时间用户情感极性分布
    user_time_senticount=[[3, 5, 13], [763, 105, 576], [876, 109, 805], [1453, 140, 1095], [5627, 497, 3144],
                          [2916, 420, 1767],[3398, 502, 3131], [7328, 940, 6149], [15219, 1579, 10897],
                          [16276, 1868, 13630], [26454, 3719, 23122]]
    user_position_senticount=[['黑龙江', 1428, 246, 1270], ['北京', 5505, 677, 4699], ['辽宁', 8121, 1290, 6342],
                              ['内蒙古', 1149, 165, 1121], ['香港', 577, 67, 431], ['天津', 1107, 138, 922],
                              ['云南', 1240, 129, 1007], ['湖南', 2125, 194, 1533],['河南', 2083, 230, 1621],
                              ['山东', 2873, 250, 2488], ['西藏', 370, 51, 438], ['广西', 1077, 131, 962],
                              ['山西', 1356, 127, 1359], ['台湾', 818, 56, 444], ['新疆', 781, 86, 666],
                              ['江西', 983, 140, 1218], ['吉林', 2066, 273, 1208], ['河北', 1928, 225, 1507],
                              ['四川', 2518, 299, 2027], ['甘肃', 1001, 131, 695], ['福建', 2465, 344, 1761],
                              ['广东', 6847, 912, 5948], ['安徽', 1425, 132, 962], ['浙江', 3947, 406, 3044],
                              ['上海', 3677, 384, 2987],['陕西', 1744, 160, 1163], ['澳门', 474, 51, 484],
                              ['海外', 4319, 628, 2467], ['江苏', 4027, 464, 3104],['湖北', 1712, 193, 1362],
                              ['海南', 697, 136, 663], ['贵州', 1025, 106, 806], ['重庆', 1275, 167, 987],
                              ['其他', 8699, 966, 6681], ['青海', 637, 57, 476], ['宁夏', 397, 24, 275]]
    user_position_senticount = json.dumps(user_position_senticount)
    user_time_senticount = json.dumps(user_time_senticount)
    user_timeciyun_path=data_path_cache+'/result/time_wc_chuo'#时间词云文件夹路径
    user_positionciyun_path=data_path_cache+'/result/position_wc_chuo'#省份词云文件夹路径
    user_timeciyun_path = json.dumps(user_timeciyun_path)
    user_positionciyun_path = json.dumps(user_positionciyun_path)
    if request.method == 'GET':
        return render_template('sentiment_detail.html', user_time_senticount=user_time_senticount,
                               user_position_senticount=user_position_senticount,user_timeciyun_path=user_timeciyun_path,
                               user_positionciyun_path=user_positionciyun_path)
#
if __name__ == '__main__':
    app.run(debug=True)
