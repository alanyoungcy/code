#coding: utf-8

import numpy as np 
import urllib
import os
import webbrowser as web

#從百度下載東西近來的
# a = np.array([1,2,5,6,6])
# b = np.array([2,4,6,7,3])
sURL = "http://www.baidu.com"
contnet =  urllib.urlopen(sURL).read()
open("baidu.html","w").write(contnet)

web.open_new_tab("baidu.html")
pic_url = "https://www.baidu.com/img/bd_logo1.png"
pic_name = os.path.basename(pic_url)
urllib.urlretrieve(pic_url,pic_name)

