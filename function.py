import re
import jieba
import jieba.analyse
import numpy
import os
import itchat
import math
import pandas
import cv2
from tqdm import tqdm
from pyecharts import Map
from wordcloud import WordCloud,ImageColorGenerator
from matplotlib import pyplot
import PIL.Image as Image
from snownlp import SnowNLP

pyplot.rcParams['font.sans-serif'] = ['SimHei']
pyplot.rcParams['axes.unicode_minus'] = False

# 获取各类数据
def get_var(friends,var):
    variable = []
    for i in friends:
        value = i[var]
        variable.append(value)
    return variable

# 数据导出为csv
def ExportAsCSV(friends):
    # 获取各类信息
    NickName = get_var(friends, "NickName")
    Sex = get_var(friends, "Sex")
    Province = get_var(friends, "Province")
    City = get_var(friends, "City")
    Signature = get_var(friends, "Signature")

    # 数据导出为csv
    data = {'NickName': NickName, 'Sex': Sex, 'Province': Province,
            'City': City, 'Signature': Signature}
    frame = pandas.DataFrame(data)
    frame.to_csv('./Resource/data.csv', index=True)

# 显示性别比例
def analyse_sex(friends):
    # 初始化计数器
    male = female = other = 0
    for i in friends[1:]:
        sex = i["Sex"]
        if sex == 1:
            male += 1
        elif sex == 2:
            female += 1
        else:
            other += 1

    # 统计好友数量
    total = len(friends[1:])
    rate_male = float(male)/total * 100.00
    rate_female = float(female)/total * 100.00
    rate_other = float(other)/total * 100.00
    num = [rate_male,rate_female,rate_other]

    # 柱状图
    pyplot.bar(range(len(num)),num,color=('red', 'yellowgreen', 'lightskyblue'))
    pyplot.title('%s的好友性别比例' % friends[0]['NickName'])
    pyplot.xlabel('性别')
    pyplot.ylabel('比例')
    pyplot.xticks((0,1,2),('男','女','不明'))
    for a, b in zip(range(3), num):
        pyplot.text(a, b + 0.05, '%.2f%%' % b, ha='center', va='bottom', fontsize=10)
    pyplot.show()

    # 饼状图
    labels = ['男','女','不明']
    explode = (0.05, 0, 0)
    pyplot.pie(num, explode=explode, labels=labels, colors=('red', 'yellowgreen', 'lightskyblue'),
             labeldistance=1.1, autopct='%2.2f%%', shadow=False,
             startangle=90, pctdistance=0.6)
    pyplot.axis('equal')
    pyplot.title('%s的好友性别比例' % friends[0]['NickName'])
    pyplot.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))
    pyplot.grid()
    pyplot.show()

# 分析省份
def analyse_province(friends):
    df = pandas.read_csv('./Resource/data.csv', encoding='gb18030')
    Province = pandas.DataFrame(df["Province"].value_counts()[:15])
    index_list = []
    for i in list(Province.index):
        if i == "":
            i = "未知"
        index_list.append(i)
    Province.index = index_list

    for i in range(len(["Province"])):
        feature = ["Province"][i]
        x_labels = list(Province.index)
        x = range(len(x_labels))
        pyplot.xticks(x, x_labels)
        y = Province[feature]
        x = [j + 0.2 * i for j in x]
        pyplot.bar(x, y, width=0.5, label="省份")

    for a, b in zip(range(15), Province.values):
        pyplot.text(a, b + 1, '%d' % b, ha='center', va='bottom', fontsize=10)

    pyplot.legend()
    pyplot.title('%s的微信好友来自的省份' % friends[0]['NickName'])
    pyplot.show()

# 分析城市
def analyse_city(friends):
    df = pandas.read_csv('./Resource/data.csv', encoding='gb18030')
    City = pandas.DataFrame(df["City"].value_counts()[:15])
    index_list = []
    for i in list(City.index):
        if i == "":
            i = "未知"
        index_list.append(i)
    City.index = index_list

    for i in range(len(["City"])):
        feature = ["City"][i]
        x_labels = list(City.index)
        x = range(len(x_labels))
        pyplot.xticks(x, x_labels)
        y = City[feature]
        x = [j + 0.2 * i for j in x]
        pyplot.bar(x, y, width=0.5, label="城市")

    for a, b in zip(range(15), City.values):
        pyplot.text(a, b + 1, '%d' % b, ha='center', va='bottom', fontsize=10)

    pyplot.legend()
    pyplot.title('%s的微信好友来自的城市' % friends[0]['NickName'])
    pyplot.show()

# 在中国地图和广东地图上显示好友数量，以html格式保存
def analyse_distribution(friends):
    users = dict(province=get_var(friends, "Province"),
                 city=get_var(friends, "City"),
                 nickname=get_var(friends, "NickName"))
    provinces = pandas.DataFrame(users)

    # 全国分布
    provinces_count = provinces.groupby('province', as_index=True)['province'].count().sort_values()
    attr_1 = list(map(lambda x: x if x != '' else '未知', list(provinces_count.index)))  # 未填写地址的改为未知
    value_1 = list(provinces_count)
    map_1 = Map("%s的微信好友全国分布图" % friends[0]['NickName'], title_pos="center", width=1200, height=600)
    map_1.add('', attr_1, value_1, is_label_show=True, is_visualmap=True, visual_text_color='#000', visual_range=[0, 120])
    map_1.render("%s的微信好友全国分布情况.html"% friends[0]['NickName'])

    # 广东分布
    citys_count = provinces.groupby('city', as_index=True)['city'].count().sort_values()
    attr_2 = list(map(lambda x: x+'市' if x != '' else '未知', list(citys_count.index)))  # 未填写地址的改为未知
    value_2 = list(citys_count)
    map_2 = Map("%s的微信好友广东分布图" % friends[0]['NickName'], title_pos="center", width=1200, height=600)
    map_2.add('', attr_2, value_2, maptype='广东', is_label_show=True, is_visualmap=True, visual_text_color='#000', visual_range=[0, 120])
    map_2.render("%s的微信好友广东分布情况.html"% friends[0]['NickName'])

# 显示个性签名,云词
def analyse_signature(friends):
    signatures = ''
    emotions = []
    pattern = re.compile("1f\d.+")
    for friend in friends:
        signature = friend['Signature']
        if(signature != None):
            signature = signature.strip().replace('span', '').replace('class', '').replace('emoji', '')
            signature = re.sub(r'1f(\d.+)','',signature)
            if(len(signature)>0):
                nlp = SnowNLP(signature)
                emotions.append(nlp.sentiments)
                signatures += ' '.join(jieba.analyse.extract_tags(signature,5))
    with open('./Resource/signatures.txt','wt',encoding='utf-8') as file:
         file.write(signatures)

    # Sinature WordCloud
    back_coloring = numpy.array(Image.open('./Resource/image.jpg'))
    wordcloud = WordCloud(
        font_path='./Resource/simfang.ttf',
        background_color="white",
        max_words=700,              # 1200
        mask=back_coloring,
        max_font_size=65,           # 75
        random_state=45,            # 45
    )

    wordcloud.generate(signatures)
    image_colors = ImageColorGenerator(back_coloring)
    pyplot.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
    pyplot.axis("off")
    pyplot.figure()
    pyplot.imshow(back_coloring, interpolation="bilinear")
    pyplot.axis("off")
    pyplot.show()
    wordcloud.to_file('./Resource/signatures.jpg')

    # 个性签名情感分析
    count_good = len(list(filter(lambda x:x>0.66,emotions)))
    count_normal = len(list(filter(lambda x:x>=0.33 and x<=0.66,emotions)))
    count_bad = len(list(filter(lambda x:x<0.33,emotions)))
    count_all = count_good + count_normal + count_bad

    rate_good = float(count_good)/count_all * 100.0
    rate_normal = float(count_normal)/count_all * 100.0
    rate_bad = float(count_bad)/count_all * 100.0

    # 柱状图
    labels = ['正面积极','中性','负面消极']
    values = (rate_good, rate_normal, rate_bad)
    pyplot.xlabel('情感判断')
    pyplot.ylabel('比例')
    pyplot.xticks(range(3),labels)
    pyplot.legend(loc='upper right')
    pyplot.bar(range(3), values, color=('red', 'yellowgreen', 'lightskyblue'))
    for a, b in zip(range(3), values):
        pyplot.text(a, b + 0.05, '%.2f%%' % b, ha='center', va='bottom', fontsize=10)
    pyplot.title('%s的微信好友签名信息情感分析' % friends[0]['NickName'])
    pyplot.show()

    # 饼状图
    pyplot.pie(values, labels=labels, colors=('red', 'yellowgreen', 'lightskyblue'),
             labeldistance=1.1, autopct='%2.2f%%', shadow=False,
             startangle=90, pctdistance=0.6)
    pyplot.axis('equal')
    pyplot.title('%s的微信好友签名信息情感分析' % friends[0]['NickName'])
    pyplot.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))
    pyplot.grid()
    pyplot.show()

# 拼接好友头像
def analyse_headimg(friends):
    num = 0
    use_face = 0
    no_use_face = 0
    if not os.path.exists('./Resource/headImg'):
        os.mkdir('./Resource/headImg')

    print("Downloading headImg......")
    for i, j in zip(friends, tqdm(range(len(friends)))):
        img = itchat.get_head_img(i["UserName"])
        with open('./Resource/headImg/' + str(num) + ".jpg", 'wb') as f:
            f.write(img)
            f.close()
            num += 1
    print("\nDownload completed !\nAnalyzing.....\nPlease wait a few seconds......")
    # 获取文件夹内的文件个数
    length = len(os.listdir('./Resource/headImg'))
    # 根据总面积求每一个的大小
    each_size = int(math.sqrt(float(600 * 600) / length))
    # 每一行可以放多少个
    lines = int(600 / each_size)
    # 生成白色背景新图片
    image = Image.new('RGBA', (600, 600), 'white')
    x = 0
    y = 0
    for i in range(0, length):
        try:
            img = Image.open('./Resource/headImg/' + str(i) + ".jpg")
            img = img.convert("RGB")
            face_img = cv2.imread('./Resource/headImg/' + str(i) + ".jpg")
            if detect_face(face_img)==True:
                use_face += 1
            else:
                no_use_face += 1
        except IOError:
            continue
            # print(i)
            # print("Error")
        else:
            img = img.resize((each_size, each_size), Image.ANTIALIAS)  # 保存图像清晰度
            image.paste(img, (x * each_size, y * each_size))
            x += 1
            if x == lines:
                x = 0
                y += 1
    image.show()

    # 饼状图
    labels = ['使用人脸头像','不使用人脸头像']
    values = (float(use_face)/length, float(no_use_face)/length)
    pyplot.pie(values, labels=labels, colors=('red', 'lightskyblue'),
             labeldistance=1.1, autopct='%2.2f%%', shadow=False,
             startangle=90, pctdistance=0.6)
    pyplot.axis('equal')
    pyplot.title('%s的微信好友使用人脸头像情况' % friends[0]['NickName'])
    pyplot.legend(loc='upper right')
    pyplot.grid()
    pyplot.show()

# 人脸检测
def detect_face(image):
    faceCascade = cv2.CascadeClassifier("./Resource/haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(5,5))
    if len(faces) > 0:
        return True
    else:
        return False