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
from wordcloud import WordCloud, ImageColorGenerator
from snownlp import sentiment
from PIL import Image, ImageDraw, ImageFont
from os import path
import math as m
import networkx as nx
import datetime

data_path = 'static/data/weibo'
data_path_cache = 'static/data/weibo/analysis_cache'


def load_wb_data(path):
    weibo = pd.read_csv(path)
    weibo['user'] = weibo['user'].map(eval)  # 将读取到的str数据转为dict类型
    num_wb = len(weibo)
    return weibo, num_wb

def load_ur_data(path):
    user = pd.read_csv(path)
    # weibo['user'] = weibo['user'].map(eval)  # 将读取到的str数据转为dict类型
    num_ur = len(user)
    return user, num_ur

class Repost():
    def __init__(self, path):
        self.reposts = self.load_repost_data(path)
        self.num_reposts = len(self.reposts)  # 转发微博总数量
        self.src_wb = self.reposts.iloc[0,]  # 源微博
        self.network = self.get_network()  # 网络结构
        self.post_indexs = {str(self.src_wb['id']): 0}  # 每条微博的编号
        self.coordinate = {str(self.src_wb['id']): {'x': 0, 'y': 0}}  # 节点坐标
        self.node_size = {str(self.src_wb['id']): 35}  # 节点大小
        self.category = {str(self.src_wb['id']): 0}  # 节点类别
        self.st_category = {}  # 节点情感类别
        self.num_category = 1  # 类别个数
        self.calc_all_node_cor()  # 计算所有节点的坐标
        self.graph_data = self.get_graph_data()  # 传给前端的图数据

    # 读取转发数据
    def load_repost_data(self, path):
        reposts = pd.read_csv(path)
        reposts['user'] = reposts['user'].map(eval)
        return reposts

    # 获取网络结构数据
    def get_network(self):
        network = []
        for i in range(1, self.num_reposts):
            post = self.reposts.iloc[i,]
            link = [str(post['pidstr']), str(post['id'])]
            if link not in network:
                network.append(link)
        return nx.DiGraph(network)

    # 获取传给前端的图数据
    def get_graph_data(self):
        nodes = [{  # 源微博节点
            # 'attributes': {'modularity_class': 0},
            'id': str(self.src_wb['id']),
            'category': self.category[str(self.src_wb['id'])],
            'itemStyle': '',
            # 'label': {'normal': {'show': 'false'}},
            'label': {'show': 'false'},
            'name': str(self.src_wb['user']['screen_name']),
            'symbolSize': self.node_size[str(self.src_wb['id'])],
            'value': self.src_wb['text'],
            'x': self.coordinate[str(self.src_wb['id'])]['x'],
            'y': self.coordinate[str(self.src_wb['id'])]['y']
        }]
        div_txt = clearTxt(str(self.src_wb['text']))
        st_cat = 2
        if div_txt:
            senti_value = sen_value(div_txt)
            if senti_value > 0.57:
                st_cat = 1
            elif senti_value < 0.5:
                st_cat = 0
        self.st_category.update({str(self.src_wb['id']):st_cat})
        links = []
        cur_nodes = []
        cur_links = []
        cur_index = 1
        for i in range(1, self.num_reposts):
            post = self.reposts.iloc[i,]  # 第i条转发微博
            node = str(post['id'])
            if node not in cur_nodes:
                self.post_indexs.update({node: cur_index})
                cur_index += 1
                # 计算微博的情感值
                div_txt = clearTxt(str(post['text']))
                st_cat = 2
                if div_txt:
                    senti_value = sen_value(div_txt)
                    if senti_value > 0.57:
                        st_cat = 1
                    elif senti_value < 0.5:
                        st_cat = 0
                self.st_category.update({node: st_cat})
                nodes.append({
                    # 'attributes': {'modularity_class': 1},
                    'id': node,
                    'category': self.category[node],
                    'itemStyle': '',
                    'label': {'normal': {'show': 'false'}},
                    'name': post['user']['screen_name'],
                    'symbolSize': self.node_size[node],
                    'value': str(post['text']),
                    'x': self.coordinate[node]['x'],
                    'y': self.coordinate[node]['y']
                })
                cur_nodes.append(node)
            link = [str(post['pidstr']), str(post['id'])]
            if link not in cur_links:
                link_id = len(links)
                links.append({
                    'id': link_id,
                    'lineStyle': {'normal': {}},
                    'name': 'null',
                    'source': link[0],
                    'target': link[1]
                })
                cur_links.append(link)



        graph_data = {
            'nodes': nodes,
            'links': links
        }
        return graph_data

    # 计算节点坐标
    def calc_all_node_cor(self):
        nodes_list = [str(self.src_wb['id'])]  # 邻居个数不为零的节点
        while len(nodes_list) != 0:
            node = nodes_list[0]
            nodes_list = self.calc_one_node_cor(node, nodes_list)
            nodes_list.pop(0)

    # 计算节点node邻居节点坐标
    def calc_one_node_cor(self, node, nodes_list):
        num_nbrs = self.network.out_degree(node)  # node的邻居节点数量
        neighbors = self.network.neighbors(node)  # node的邻居节点
        if self.coordinate.get(node):
            node_x = self.coordinate[node]['x']  # node的x坐标
            node_y = self.coordinate[node]['y']  # node的y坐标
        else:
            print('节点{}的坐标不存在'.format(node))
            return nodes_list
        i = 0
        j = 0
        for nbr in neighbors:
            nbr_out = self.network.out_degree(nbr)
            if nbr_out > 0:
                nodes_list.append(nbr)  # nbr节点邻居个数不为零，加入到nodes_list中

            # 计算nbr到父节点的半径r和节点大小
            r = 1.0
            size = 15
            category = self.category[node]
            if num_nbrs < 10:
                r = 0.5 * r
            elif num_nbrs < 400:
                r = r
            else:
                r = 1.5 * r
            if nbr_out > 1 and nbr_out < 10:
                r = 2 * r
                category = self.num_category
                self.num_category += 1
            elif nbr_out >= 10 and nbr_out < 100:
                r = 2.2 * r
                size = size + nbr_out / 10
                category = self.num_category
                self.num_category += 1
            elif nbr_out >= 100:
                r = 2.5 * r
                size = size + 10 + nbr_out / 100
                category = self.num_category
                self.num_category += 1

            # 计算节点坐标
            if num_nbrs == 1:  # 父节点只有一个出边邻居
                # 计算父节点的父节点坐标
                pro_node = next(self.network.predecessors(node))
                pro_node_x = self.coordinate[pro_node]['x']
                pro_node_y = self.coordinate[pro_node]['y']
                nbr_x = node_x + (node_x - pro_node_x) * 0.7
                nbr_y = node_y + (node_y - pro_node_y) * 0.7
                if nbr not in self.coordinate:
                    self.coordinate.update({nbr: {'x': nbr_x, 'y': nbr_y}})
                    self.node_size.update({nbr: size})
                    self.category.update({nbr: category})
            else:
                nbr_x = r * m.cos(i * 2 * m.pi / num_nbrs)
                if nbr_out >= 10:
                    nbr_x = r * m.cos(m.pi / 4)
                    j += 1

                nbr_y = 0
                if i < num_nbrs / 2:
                    nbr_y += m.sqrt(r ** 2 - nbr_x ** 2)
                else:
                    nbr_y -= m.sqrt(r ** 2 - nbr_x ** 2)
                if nbr not in self.coordinate:
                    self.coordinate.update({nbr: {'x': node_x + nbr_x, 'y': node_y + nbr_y}})
                    self.node_size.update({nbr: size})
                    self.category.update({nbr: category})
                i += 1
        return nodes_list


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
                     open(data_path_cache + '/stopwords.txt', encoding='gbk').readlines()]
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
    weibo_80k_data = pd.read_csv(data_path + '/weibo_senti_80k.csv', encoding='utf-8')
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
    neg_file = open(data_path_cache + '/neg_file.txt', 'w+', encoding='utf-8')
    pos_file = open(data_path_cache + '/pos_file.txt', 'w+', encoding='utf-8')
    for i in range(len(weibo_80k_data)):
        if y[i] == 0:
            neg_file.write(clearTxt(x[i]) + '\n')
        else:
            pos_file.write(clearTxt(x[i]) + '\n')
    sentiment.train(data_path_cache + '/neg_file.txt',
                    data_path_cache + '/pos_file.txt')  # 训练语料库
    # 保存模型
    sentiment.save(data_path_cache + '/sentiment_snownlp.marshal')


# 求文本的情感倾向值，>0.57则默认为积极，<0.5则默认为消极，0.57与0.5之间可默认为中性
def sen_value(text):
    senti = sn.SnowNLP(text)
    senti_value = round((senti.sentiments), 2)
    return senti_value


# 统计所有文本情感极性，并生成词云图，不考虑时间、省份
def senti_diffusion():
    data = pd.read_csv(data_path + '/weibo.csv', encoding='utf-8')
    content = data.iloc[:, 2].values
    count = [0, 0, 0]  # 统计极性，积极，中性，消极
    for i in range(len(data)):
        data.iloc[i, 2] = clearTxt(str(data.iloc[i, 2]))  # 处理文本
        if data.iloc[i, 2] == '':
            data.iloc[i, 2] = '年月日'  # 即默认为中性态度
        senti_value = sen_value(data.iloc[i, 2])
        if (senti_value >= 0.57):
            count[0] += 1
        elif (senti_value <= 0.5):
            count[2] += 1
        elif (senti_value > 0.5 and senti_value < 0.57):
            count[1] += 1
    print(count)
    # 统计结果：[114092, 10052, 65256]
    # 生成词云
    text = ''
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
        stopwords={'网页', '链接', '博转发'},  # 设置停用词，停用词则不再词云图中表示
        max_font_size=80,  # 设置字体最大值
        random_state=5,  # 设置有多少种随机生成状态，即有多少种配色方案
        scale=1,  # 设置生成的词云图的大小
    ).generate(text)  # 生成词云
    image_colors = ImageColorGenerator(backgroud_image)
    plt.imshow(wc.recolor(color_func=image_colors))  # 显示词云图
    plt.axis('off')  # 不显示x,y轴坐

    wc.to_file(path.join(data_path_cache + '/weibo_ciyun.png'))
    return count


# 根据时间信息判断所有用户的情感倾向，生成词云图
def senti_diffusion_time():
    time_data = pd.read_excel(data_path_cache + '/weibo_time.xlsx', encoding='utf-8', sheet_name=None)
    sheet_name = time_data.keys()  # 获取当前表格所有sheet名字
    year_count = []  # 保存每一年的积极、消极以及中性文本数目
    for i in sheet_name:
        time_data_y = time_data[i]  # 当前某一年的所有用户数据
        positive = 0
        negative = 0
        neutral = 0
        wc_text = ''  # 当前年份的词云文本

        # 根据年份统计文本情感倾向
        for j in range(len(time_data_y)):
            time_data_y.iloc[j, 2] = clearTxt(str(time_data_y.iloc[j, 2]))  # 由于部分数据只有数字或者符号，需要转成字符型
            if time_data_y.iloc[j, 2] == '':
                time_data_y.iloc[j, 2] = '年月日'  # 即默认为中性态度
            senti_value = sen_value(time_data_y.iloc[j, 2])  # time_data_y.iloc[j,2]表示表格中第三列文本数据
            if (senti_value >= 0.57):
                positive += 1
            elif (senti_value <= 0.5):
                negative += 1
            elif (senti_value > 0.5 and senti_value < 0.57):
                neutral += 1
        year_count.append([positive, neutral, negative])  # 保存每一年的积极、消极以及中性文本数目
        # [[3, 5, 13], [763, 105, 576], [876, 109, 805], [1453, 140, 1095], [5627, 497, 3144], [2916, 420, 1767],
        #  [3398, 502, 3131], [7328, 940, 6149], [15219, 1579, 10897], [16276, 1868, 13630], [26454, 3719, 23122]]

        # 根据年份文本形成词云图
        for j in range(len(time_data_y)):
            line = re.sub("[\s+\.\!\/_,$%^*(+]\"\']+|[+——！，:\[\]。!：？?～”,、\.\/~@#￥%……&*【】 （）]+", "",
                          str(time_data_y.iloc[j, 2]))  # 去除符号
            wc_text += ' '.join(jieba.cut(line, cut_all=True))
        # 设置词云格式
        backgroud_image = np.array(Image.open(r"static/image/wc_time.jpg"))
        wc_year = WordCloud(
            background_color='white',  # 设置背景颜色，与图片的背景色相关
            mask=backgroud_image,  # 设置背景图片
            font_path='C:\Windows\Fonts\STZHONGS.TTF',  # 显示中文，可以更换字体
            max_words=300,  # 设置最大显示的字数
            stopwords={'网页', '链接'},  # 设置停用词，停用词则不再词云图中表示
            max_font_size=80,  # 设置字体最大值
            random_state=5,  # 设置有多少种随机生成状态，即有多少种配色方案
            scale=1,  # 设置生成的词云图的大小
        ).generate(wc_text)  # 生成词云
        image_colors = ImageColorGenerator(backgroud_image)
        plt.imshow(wc_year.recolor(color_func=image_colors))  # 显示词云图
        # plt.imshow(wc_year, interpolation='bilinear')
        plt.axis('off')  # 不显示x,y轴坐
        # 按递增顺序保存生成的词云图

        wc_year.to_file(path.join(data_path_cache + '/time_wc', str(i) + '_wc.png'))

    # 生成动态gif文件
    def create_gif(image_list, gif_name):
        frames = []
        for image_name in image_list:
            frames.append(imageio.imread(image_name))
        # 保存gif文件
        imageio.mimsave(gif_name, frames, 'gif', duration=0.3)
        return

    def find_all_gif():

        png_filenames = glob.glob(data_path_cache + '/result/time_wc_chuo/*')  # 加入图片位置，绝对路径
        buf = []
        for png_file in png_filenames:
            buf.append(png_file)
        return buf

    # 为每张图片生成时间戳
    png_list = os.listdir(data_path_cache + '/time_wc')
    font = ImageFont.truetype("C:\Windows\Fonts\STZHONGS.TTF", 40)
    j = 0
    dir_list = os.listdir(data_path_cache + '/time_wc/')
    dir = []

    for i in dir_list:
        dir.append(re.sub('_wc', '', str(os.path.splitext(i)[0])))
    for png in png_list:
        name = png
        imageFile = data_path_cache + '/time_wc/' + name
        im = Image.open(imageFile)
        draw = ImageDraw.Draw(im)
        draw.text((0, 0),  # 设置字体位置
                  dir[j] + '年',  # 设置内容
                  (0, 0, 0),  # 设置颜色
                  font=font)  # "设置字体
        draw = ImageDraw.Draw(im)
        # 存储图片
        im.save(data_path_cache + '/result/time_wc_chuo/' + name)
        j += 1
    buff = find_all_gif()
    time_wc = create_gif(buff, data_path_cache + '/result/time_wc.gif')  # 生成时间词云动态gif图
    return year_count, time_wc


# 根据地域信息判断所有用户的情感倾向，生成词云图
def senti_diffusion_position():
    position_data = pd.read_excel(data_path_cache + '/weibo_position.xlsx', encoding='utf-8', sheet_name=None)
    sheet_name = position_data.keys()  # 获取当前表格所有sheet名字
    # print(sheet_name)
    position_count = []  # 保存每个省份的积极、消极以及中性文本数目
    for i in sheet_name:
        position_data_y = position_data[i]  # 当前某省份的所有用户数据
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
        position_count.append([i, positive, neutral, negative])  # 保存每个省份的积极、消极以及中性文本数目
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
            scale=1  # 设置生成的词云图的大小
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
        position_n = os.path.splitext(i)[0]
        if position_n == '黑龙_wc':
            position_n = '黑龙江_wc'
        if position_n == '内蒙_wc':
            position_n = '内蒙古_wc'
        dir.append(re.sub('_wc', '', str(position_n)))
    for png in png_list:
        name = png
        imageFile = data_path_cache + '/position_wc/' + name
        im = Image.open(imageFile)
        draw = ImageDraw.Draw(im)
        draw.text((0, 0),  # 设置字体位置
                  dir[j],  # 设置内容
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
user, num_ur = load_ur_data('static/data/weibo/user.csv')
repost = Repost('static/data/weibo/repost.csv')


@app.route('/')
def hello_world():
    return render_template('index.html')


# 微博数据可视化
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

    # 微博数量-时间
    years = []
    for i in weibo['created_at']:
        date = i.split('-')
        years.append(date[0])
    year_count = pd.Series(years).value_counts().sort_index(ascending=False)
    '''year_count = {}
    for i in range(len(year_list)):
        year = year_list.index[i]
        num = year_list[i]
        year_count.update({year: int(num)})'''
    data['year_count'] = year_count.to_dict()

     # 转发微博占比
    repo_count = 0
    post_count = 0
    for i in range(num_wb):
        if not pd.isnull(weibo.loc[i]['retweeted_status']):
            repo_count += 1
        elif pd.isnull(weibo.loc[i]['retweeted_status']):
            post_count += 1
    repost_rate = {"转发微博": int(repo_count)}
    repost_rate.update({"原创微博": int(post_count)})
    data['repost_rate'] = repost_rate

    # 微博转发、评论、点赞分布
    reposts_count = weibo['reposts_count'].value_counts().sort_index()
    comments_count = weibo['comments_count'].value_counts().sort_index() #数量统计，按index排序(为了方便嘻嘻嘻）
    attitudes_count = weibo['attitudes_count'].value_counts().sort_index()

    repo_dic = {'没有转发': int(reposts_count[0]),
                '1次转发': int(reposts_count[1]),
                '2次转发': int(reposts_count[2]),
                '3次转发': int(reposts_count[3]),
                '4次转发': int(reposts_count[4]),
                '5次转发': int(reposts_count[5]),
                '6-10次转发': int(reposts_count[6:11].sum()),
                '10次以上转发': int(reposts_count[11:].sum())}
    print(repo_dic)
    comments_dic = {'没有评论': int(comments_count[0]),
                    '1条评论': int(comments_count[1]),
                    '2条评论': int(comments_count[2]),
                    '3条评论': int(comments_count[3]),
                    '4条评论': int(comments_count[4]),
                    '5条评论': int(comments_count[5]),
                    '6-10条评论': int(comments_count[6:11].sum()),
                    '10条以上评论': int(comments_count[11:].sum())}
    print(comments_dic)
    attitudes_dic = {'没有点赞': int(attitudes_count[0]),
                     '1个点赞': int(attitudes_count[1]),
                     '2个点赞': int(attitudes_count[2]),
                     '3个点赞': int(attitudes_count[3]),
                     '4个点赞': int(attitudes_count[4]),
                     '5个点赞': int(attitudes_count[5]),
                     '6-10个点赞': int(attitudes_count[6:11].sum()),
                     '10个以上点赞': int(attitudes_count[11:].sum())}
    print(attitudes_dic)
    distribution = {
        'repo_dic': repo_dic,
        'comments_dic': comments_dic,
        'attitudes_dic': attitudes_dic
    }
    data['distribution'] = distribution

    data_json = json.dumps(data)
    return render_template('wbstatistic.html', weibo_json=data_json)

# 用户数据可视化
@app.route('/urstatistic')
def ur_statistic():
    data = {}  #所有要传给前端的数据
    # 统计转发微博的用户的性别比例
    f_count = 0
    m_count = 0
    for i in range(num_ur):
        if user.loc[i]['gender'] == 'f':
            f_count += 1
        elif user.loc[i]['gender'] == 'm':
            m_count += 1
    gender_rate = {"female": int(f_count) / num_ur}
    gender_rate.update({"male": int(m_count) / num_ur})
# 传给前端的转发微博用户性别比例数据
    gender = {
        "num_ur": int(num_ur),
        "gender_rate": gender_rate
    }
    data['gender'] = gender

# 统计转发微博用户认证比例
    verified_count = 0
    unVerified_count = 0
    for i in range(num_ur):
        if user.loc[i]['verified']:
            verified_count += 1
        elif not user.loc[i]['verified']:
            unVerified_count += 1
    verified_rate = {"认证": int(verified_count) / num_ur}
    verified_rate.update({"非认证": int(unVerified_count) / num_ur})
    verified = {
        "verified_rate": verified_rate
    }
    data['verified'] = verified
#用户粉丝数分布
    followers_count = user['followers_count'].value_counts().sort_index()
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    sum5 = 0
    for i in range(len(followers_count)):
        if followers_count.index[i] < 50:
            sum1 += followers_count.iloc[i]
        elif followers_count.index[i] >= 50 and followers_count.index[i] < 150:
            sum2 += followers_count.iloc[i]
        elif followers_count.index[i] >= 150 and followers_count.index[i] < 300:
            sum3 += followers_count.iloc[i]
        elif followers_count.index[i] >= 300 and followers_count.index[i] < 500:
            sum4 += followers_count.iloc[i]
        elif followers_count.index[i] >= 500:
            sum5 += followers_count.iloc[i]
    followers_dic = {'50以下': int(sum1),
                     '50-149': int(sum2),
                     '150-299': int(sum3),
                     '300-499': int(sum4),
                     '500及以上': int(sum5)}
    data['followers_dic'] = followers_dic
#用户关注数分布
    follow_count = user['follow_count'].value_counts().sort_index()
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    sum5 = 0
    sum6 = 0
    for i in range(len(follow_count)):
        if follow_count.index[i] < 200:
            sum1 += follow_count.iloc[i]
        elif follow_count.index[i] >= 200 and follow_count.index[i] < 500:
            sum2 += follow_count.iloc[i]
        elif follow_count.index[i] >= 500 and follow_count.index[i] < 1000:
            sum3 += follow_count.iloc[i]
        elif follow_count.index[i] >= 1000 and follow_count.index[i] < 2000:
            sum4 += follow_count.iloc[i]
        elif follow_count.index[i] >= 2000 and follow_count.index[i] < 5000:
            sum5 += follow_count.iloc[i]
        elif follow_count.index[i] >= 5000:
            sum6 += follow_count.iloc[i]
    follow_dic = {'200以下': int(sum1),
                  '200-499': int(sum2),
                  '500-999': int(sum3),
                  '1000-1999': int(sum4),
                  '2000-4999': int(sum5),
                  '5000及以上': int(sum6)}
    data['follow_dic'] = follow_dic
#用户微博数分布
    statuses_count = user['statuses_count'].value_counts().sort_index()
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    sum5 = 0
    sum6 = 0
    for i in range(len(statuses_count)):
        if statuses_count.index[i] < 500:
            sum1 += statuses_count.iloc[i]
        elif statuses_count.index[i] >= 500 and statuses_count.index[i] <= 1000:
            sum2 += statuses_count.iloc[i]
        elif statuses_count.index[i] >= 1000 and statuses_count.index[i] <= 3000:
            sum3 += statuses_count.iloc[i]
        elif statuses_count.index[i] >= 3000 and statuses_count.index[i] <= 5000:
            sum4 += statuses_count.iloc[i]
        elif statuses_count.index[i] >= 5000 and statuses_count.index[i] <= 10000:
            sum5 += statuses_count.iloc[i]
        elif statuses_count.index[i] >= 10000:
            sum6 += statuses_count.iloc[i]
    statuses_dic = {'500以下': int(sum1),
                    '500-999': int(sum2),
                    '1000-2999': int(sum3),
                    '3000-4999': int(sum4),
                    '5000-9999': int(sum5),
                    '10000以上': int(sum6)}
    data['statuses_dic'] = statuses_dic
#用户所在地分布
    province = []
    for i in user['position']:
        pos = i.split(' ')
        province.append(pos[0])
    province_list = pd.Series(province).value_counts()
    data['province_list'] = province_list.to_dict()
    data_json = json.dumps(data)
    return render_template('urstatistic.html', data_json=data_json)


# 微博转发结构页面数据
@app.route('/rpstructure')
def wb_forward():
    data = {'graph_data': repost.graph_data,
            'node_num': repost.num_reposts,
            'num_category': repost.num_category,
            'category': repost.category,
            'st_category': repost.st_category}  # 所有要传给前端的数据

    # 统计不同时间的转发微博
    times = repost.reposts.sort_values(by='created_at')['created_at'].drop_duplicates().to_list()  # 所有时间集合
    rp_records = []  # 记录每个时间段内的微博编号
    for t in times:
        posts_id = repost.reposts[repost.reposts['created_at'] == t]['id'].to_list()  # t时刻转发的微博ID
        # 将微博id转换为微博编号
        post_indexs = []
        for post_id in posts_id:
            post_indexs.append(repost.post_indexs[str(post_id)])
        rp_records.append({'time': t, 'post_indexs': post_indexs})
    data['rp_records'] = rp_records

    weiboRepost = repost.reposts
    num_repost = repost.num_reposts
    users = weiboRepost.loc[:, 'user']
    # 统计转发微博的用户的性别比例
    f_count = 0
    m_count = 0
    for i in range(1, num_repost):  # 第一条微博是原微博，不计入统计
        if users[i]['gender'] == 'f':
            f_count += 1
        elif users[i]['gender'] == 'm':
            m_count += 1
    gender_rate = {"female": int(f_count) / num_repost}
    gender_rate.update({"male": int(m_count) / num_repost})
    # 传给前端的转发微博用户性别比例数据
    gender = {
        "num_wb": int(num_wb),
        "gender_rate": gender_rate
    }
    data['gender'] = gender

    # 统计转发微博用户认证比例
    verified_count = 0
    unVerified_count = 0
    for i in range(1, num_repost):
        if users[i]['verified']:
            verified_count += 1
        elif not users[i]['verified']:
            unVerified_count += 1
    verified_rate = {"认证": int(verified_count) / num_repost}
    verified_rate.update({"非认证": int(unVerified_count) / num_repost})
    verified = {
        "verified_rate": verified_rate
    }
    data['verified'] = verified

    # 转发层级
    startP = repost.src_wb  # 源微博
    graph = repost.network  # 关系图
    nodes_list = list(graph.neighbors(str(startP[2])))  # 源微博的邻居节点，也就是一级转发
    level_num = 0  # 级数
    level_dic = []  # 级数-数量
    visited = [str(startP[2])]  # 已经算过的节点
    level_rate = []  # 每一级微博数量占所有微博的比例

    while len(nodes_list) != 0:  # 类似于广度遍历
        length = len(nodes_list)
        level_num += 1
        if level_num == 1:
            level_dic = {'1': str(length)}
            level_rate = {'1': length / num_repost}
        else:
            level_dic.update({str(level_num): str(length)})
            level_rate.update({str(level_num): length / num_repost})
        new_neigh_list = []  # 下一级节点
        for i in range(length):
            now_node_list = list(graph.neighbors(nodes_list[i]))  # 当前节点的邻居节点
            for j in range(len(now_node_list)):  # 遍历新的邻居节点
                if visited.count(now_node_list[j]):  # 已经遍历过该节点
                    break
                visited.append(now_node_list[j])  # 加入到已遍历节点
                new_neigh_list.append(now_node_list[j])
        nodes_list = new_neigh_list
    level = {
        "level_dic": level_dic,
        "level_rate": level_rate,
    }
    data['level'] = level

    # 转发量-时间
    time_count = weiboRepost.loc[:, 'created_at'].value_counts()
    startP = weiboRepost.loc[0]
    startTime = startP['created_at'].split("-")
    d1 = datetime.datetime(int(startTime[0]), int(startTime[1]), int(startTime[2]))
    max_day = 0
    for i in range(len(time_count)):
        nowTime = time_count.index[i].split("-")
        d2 = datetime.datetime(int(nowTime[0]), int(nowTime[1]), int(nowTime[2]))
        interval = d2 - d1  # 两日期差距
        if interval.days > max_day:
            max_day = interval.days
        if i == 0:
            time_repost1 = {interval.days: time_count[i]}
        else:
            time_repost1.update({interval.days: time_count[i]})
    for i in range(max_day + 1):
        if i in time_repost1.keys():
            if i == 0:
                time_repost = {str(i): (time_repost1[i] - 1) / 1.0}
            else:
                time_repost.update({str(i): time_repost1[i] / 1.0})
        else:
            if i == 0:
                time_repost = {str(i): 0 / 1.0}
            else:
                time_repost.update({str(i): 0 / 1.0})
    time = {
        "time_repost": time_repost,
    }
    data['time'] = time

    data_json = json.dumps(data)
    return render_template('rpstructure.html', data_json=data_json)


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

# 单条微博语句展示，以及情感极性测试
@app.route('/singlewb_sentiment', methods=['GET', 'POST'])
def sentiwb_test():
    weibo = pd.read_csv(data_path + '/weibo.csv', encoding='utf-8')

    global show_weibo_text
    global senti_value
    global content_err
    if request.method == 'POST':
        content = int(request.values.get('content_value'))
        if content != '':
            show_weibo_text=weibo.loc[content][2]
            content_err = 1
            senti_value = sen_value(clearTxt(show_weibo_text))
        else:
            senti_value = 0
            content_err = 0
        senti_value = json.dumps(senti_value)
        content_err = json.dumps(content_err)
        return render_template('singlewb_sentiment.html', senti_value=senti_value, content_err=content_err,show_weibo_text=show_weibo_text)
    else:
        return render_template('singlewb_sentiment.html')

# 所有数据情感分析
@app.route('/weibo_sentiment_analysis', methods=['GET', 'POST'])
def sentiment_analysis():
    with open(data_path_cache + "/result/user_position_count.txt", "r",
              encoding='utf-8') as user_position_count:
        data = user_position_count.readlines()
        user_position = []  # 用户省份地址
        user_position_c = []  # 省份用户人数
        user_text_count = [114092, 10052, 65256]  # 用户微博极性统计，根据senti_diffusion()获得
        for i in range(len(data)):
            user_position.append(data[i][0:2])
            user_position_c.append(int(data[i][3:]))

    def getresult(path):
        user_senticount = pd.read_table(path)
        return user_senticount

    if request.method == 'GET':
        user_position = json.dumps(user_position)
        user_position_c = json.dumps(user_position_c)
        user_text_count = json.dumps(user_text_count)
        user_ciyun_path = data_path_cache + '/weibo_ciyun.png'
        return render_template('weibo_sentiment_analysis.html', user_position_c=user_position_c,
                               user_position=user_position, user_text_count=user_text_count,
                               user_ciyun_path=user_ciyun_path)


# 根据时间以及省份给出极性分布及词云图
@app.route('/sentiment_detail', methods=['GET', 'POST'])
def sentiment_analysis_detail():
    # 按照时间用户情感极性分布
    user_time_senticount = [[3, 5, 13], [763, 105, 576], [876, 109, 805], [1453, 140, 1095], [5627, 497, 3144],
                            [2916, 420, 1767], [3398, 502, 3131], [7328, 940, 6149], [15219, 1579, 10897],
                            [16276, 1868, 13630], [26454, 3719, 23122]]
    user_position_senticount = [['黑龙江', 1428, 246, 1270], ['北京', 5505, 677, 4699], ['辽宁', 8121, 1290, 6342],
                                ['内蒙古', 1149, 165, 1121], ['香港', 577, 67, 431], ['天津', 1107, 138, 922],
                                ['云南', 1240, 129, 1007], ['湖南', 2125, 194, 1533], ['河南', 2083, 230, 1621],
                                ['山东', 2873, 250, 2488], ['西藏', 370, 51, 438], ['广西', 1077, 131, 962],
                                ['山西', 1356, 127, 1359], ['台湾', 818, 56, 444], ['新疆', 781, 86, 666],
                                ['江西', 983, 140, 1218], ['吉林', 2066, 273, 1208], ['河北', 1928, 225, 1507],
                                ['四川', 2518, 299, 2027], ['甘肃', 1001, 131, 695], ['福建', 2465, 344, 1761],
                                ['广东', 6847, 912, 5948], ['安徽', 1425, 132, 962], ['浙江', 3947, 406, 3044],
                                ['上海', 3677, 384, 2987], ['陕西', 1744, 160, 1163], ['澳门', 474, 51, 484],
                                ['海外', 4319, 628, 2467], ['江苏', 4027, 464, 3104], ['湖北', 1712, 193, 1362],
                                ['海南', 697, 136, 663], ['贵州', 1025, 106, 806], ['重庆', 1275, 167, 987],
                                ['其他', 8699, 966, 6681], ['青海', 637, 57, 476], ['宁夏', 397, 24, 275]]
    user_position_senticount = json.dumps(user_position_senticount)
    user_time_senticount = json.dumps(user_time_senticount)
    user_timeciyun_path = data_path_cache + '/result/time_wc_chuo'  # 时间词云文件夹路径
    user_positionciyun_path = data_path_cache + '/result/position_wc_chuo'  # 省份词云文件夹路径
    user_timeciyun_path = json.dumps(user_timeciyun_path)
    user_positionciyun_path = json.dumps(user_positionciyun_path)
    if request.method == 'GET':
        return render_template('sentiment_detail.html', user_time_senticount=user_time_senticount,
                               user_position_senticount=user_position_senticount,
                               user_timeciyun_path=user_timeciyun_path,
                               user_positionciyun_path=user_positionciyun_path)

#
if __name__ == '__main__':
    app.run(debug=True)
