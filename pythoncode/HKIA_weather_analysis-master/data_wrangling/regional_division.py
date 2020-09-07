
#  Aothor: Ethan Lee
#  E-mail: ethan2lee.coder@gmail.com


import numpy as np
import pandas as pd

def number_encoding(alpha,start,stop,step=1):
    """
    机位编号数列构造函数
    :param alpha: 机位起始字母如 N ,S ,E ,string
    :param start: 起始编号
    :param stop: 终止编号
    :param step: 间隔
    :return: 编号数列
    """
    number=np.arange(start=start,step=step,stop=stop+1,dtype="int").tolist()
    final_list=list(map(lambda x: "{}{}".format(alpha,x),number))
    return final_list


def invert_dict(d):
    """
    将字典键值对反转
    :param d:  原始字典
    :return: 反转后的字典
    """
    inverse = dict()
    for key in d:
        for item in d[key]:
            inverse[item] = key
    return inverse

# 北客运停机坪编码
north_passenger_apron=[]
north_passenger_apron.extend(number_encoding("E",15,19,2))
north_passenger_apron.extend(number_encoding("N",20,36,2))
north_passenger_apron.extend(number_encoding("N",60,70,2))
north_passenger_apron.extend(number_encoding("N",501,510))
north_passenger_apron.extend(number_encoding("N",141,141))

# 南客运停机坪编码
south_passenger_apron=[]
south_passenger_apron.extend(number_encoding("E",1,4))
south_passenger_apron.extend(number_encoding("S",21,35,2))
south_passenger_apron.extend(number_encoding("S",101,111))

# 西客运停机坪编码
west_passenger_apron=[]
west_passenger_apron.extend(number_encoding("W",40,50,2))
west_passenger_apron.extend(number_encoding("W",61,71,2))
west_passenger_apron.extend(number_encoding("W",121,126,2))
west_passenger_apron.extend(number_encoding("TW",1,2))
west_passenger_apron.extend(number_encoding("V",40,50,2))

# 中场客运停机坪编码
mindle_passenger_apron=[]
mindle_passenger_apron.extend(number_encoding("D",201,219))
mindle_passenger_apron.append("D300")
mindle_passenger_apron.extend(number_encoding("L",411,416))
mindle_passenger_apron.extend(number_encoding("L",221,226))
mindle_passenger_apron.extend(number_encoding("D",301,319))
mindle_passenger_apron.extend(number_encoding("D",321,322))

# 西货运停机坪编码
West_freight_apron=[]
West_freight_apron.extend(number_encoding("X",451,459))

# 货运停机坪编码
freight_apron=[]
freight_apron.extend(number_encoding("C",1,35))
freight_apron.append("TC1")

# 维修停机坪编码
maintenance_apron=[]
maintenance_apron.extend(number_encoding("M",1,10))
maintenance_apron.append("M1A")
maintenance_apron.extend(number_encoding("M",21,27))
maintenance_apron.extend(number_encoding("M",31,38))
maintenance_apron.extend(number_encoding("NB",21,40))

region = dict(
    {"north_passenger_apron":north_passenger_apron,
      "south_passenger_apron":south_passenger_apron,
      "west_passenger_apron":west_passenger_apron,
      "mindle_passenger_apron":mindle_passenger_apron,
      "West_freight_apron":West_freight_apron,
      "freight_apron":freight_apron,
      "maintenance_apron":maintenance_apron
      })

number_to_region=invert_dict(region)


run_ways=['07L','07R','25L','25R']

df=pd.DataFrame.from_dict(number_to_region,orient="index")
pass
