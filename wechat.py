import itchat
from function import *

# 登陆微信
itchat.login()

# 爬取好友信息
friends = itchat.get_friends(update=True)

# 数据导出为csv
ExportAsCSV(friends)

# 分析数据
analyse_sex(friends)                # 性别比例
analyse_province(friends)           # 分析省份
analyse_city(friends)               # 分析城市
analyse_distribution(friends)       # 在中国地图和广东地图上显示好友数量，以html格式保存
analyse_signature(friends)          # 个性签名分析(词云 + 情绪分析)
analyse_headimg(friends)            # 拼接头像并分析是否使用人脸头像
