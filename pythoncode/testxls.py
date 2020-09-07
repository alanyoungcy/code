
import xlwt,xlrd,xlutils
import webbrowser
import openpyxl
import web 
import json
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
 
import csv
 
pd.plotting.register_matplotlib_converters()

insurance_data = pd.read_csv('data/insurance.csv')

sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
# urls = (
#     #访问index的时候，执行index方法
#     '/index(.*)', 'index',
# )

# with open('insurance.csv') as csvfile:
#     insurance = csv.DictReader(csvfile)
#     charges = 0.0
#     for row in insurance:
#        charges =   float(row['charges']) + charges
#     print(charges)




