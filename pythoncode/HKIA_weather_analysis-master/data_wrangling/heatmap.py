# coding: utf-8
# ! /usr/local/bin/python3
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from data_wrangling.aircraft_position import a_p
from data_wrangling.read_sql import FIRST_STAND_NUMBER_ARR_DEP_RUN_WAY


def heapmap_plot(df, airflight_position_column, data_column):
    gradient = 20
    R = np.linspace(start=14, stop=252, num=gradient, dtype=int)
    G = np.linspace(start=20, stop=4, num=gradient, dtype=int)
    B = np.linspace(start=247, stop=23, num=gradient, dtype=int)
    color_bar = []
    for r, g, b in zip(R, G, B):
        color_bar.append((r, g, b))
    # color_bar = color_bar[::-1]
    df[data_column]
    max = df[data_column].max()
    min = df[data_column].min()
    position = \
        pd.merge(df, a_p, how='inner', left_on=airflight_position_column, right_on="location").set_index(
            "location")[["map_x", "map_y", data_column]].astype("int", copy=True).to_dict("index")
    # get an image
    base = Image.open('./imgs/HKIA_MAP.png').convert('RGBA')
    # color = color_bar[0]  ## (R,G,B)
    # block = Image.new('RGBA', (20, 10), color)
    txt = Image.new('RGBA', base.size, (255, 255, 255, 0))

    # get a font
    # fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
    # get a drawing context
    d = ImageDraw.Draw(txt)

    gap = np.linspace(start=1500, stop=2000, num=gradient, dtype=int)
    coordinate = (100, 100)  ##(x,y)

    for coor, c in zip(gap, color_bar):
        block = Image.new('RGBA', (60, 40), c)
        base.paste(block, (coor, 800))

    d.text((gap[gradient - 1] + 50, 780), "max", fill=(54, 45, 24))
    d.text((gap[0], 780), "min", fill=(54, 45, 24,), )

    for key, value in position.items():
        x = value['map_x']
        y = value['map_y']
        data = value[data_column]
        normalize = (data - min) / (max - min)
        index = int(normalize * (gradient - 1))
        block = Image.new('RGBA', (27, 22), color_bar[index])
        base.paste(block, (x, y))
        # make a blank image for the text, initialized to transparent text color
        # draw text, half opacity
        d.text((x, y - 10), key, fill=(54, 45, 24))
        # draw text, full
        # d.text((x, y ), str(value[data_column]), fill=(54, 45, 24))

    # base.paste(block,(500,850))
    out = Image.alpha_composite(base, txt)
    out.save("./imgs/{}/{}.png".format(airflight_position_column, data_column))
    # out.show()


if __name__ == "__main__":

    # # 生成各个天气状态下，不同跑道状态，原始航班安排位置热力图
    # for key in FIRST_STAND_NUMBER_RUN_WAYS.keys():
    #     heapmap_plot(FIRST_STAND_NUMBER_RUN_WAYS[key],"FIRST_STAND_NUMBER",key)
    #     print(key)
    # 生成各个天气状态下原始航班安排位置热力图
    for key in FIRST_STAND_NUMBER_ARR_DEP_RUN_WAY.keys():
        heapmap_plot(FIRST_STAND_NUMBER_ARR_DEP_RUN_WAY[key], "FIRST_STAND_NUMBER", key)
        print(key)
