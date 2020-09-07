#coding:utf-8

import numpy as np
# import pandas as pd
# import seaborn as sns
# import sklearn as sk
# import matplotlib as mpl
import matplotlib.pyplot as plt
# import os  
# #加載Iris的數據庫
#iris = datasets.load_iris()

# print(iris['data'])
# print(iris['target'])
# print(iris['target_names'])
# print(iris['DESCR'])
# print(iris['feature_names'])

# print(np.unique(iris['target']))

baseDir = "/Users/alanyoung/Documents/code/pythoncode/"
csvDir="horse-racing-dataset-for-experts-hong-kong/"
np.random.seed(1000)
y = np.random.standard_normal((1000, 2))
plt.figure(figsize=(7, 5))
plt.plot(y[:, 0], y[:, 1], 'ro')
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Scatter Plot')
# plt.show()

# horseresult = pd.read_csv(baseDir+csvDir+"results.csv")
# print(horseresult.dtypes)

# sns.set(rc={'figure.figsize':(16.7,13.27)})
# sns.violinplot(x="raceno",y="distance", hue="raceno", data=horseresult)
# plt.show()
