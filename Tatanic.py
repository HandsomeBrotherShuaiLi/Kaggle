#coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import scipy
from pandas import Series,DataFrame
data_train=pd.read_csv("S:/李帅的技能书/train.csv")
print(data_train.info())
print(data_train.describe())
fig = plt.figure()
fig.set(alpha=0.2)
plt.subplot2grid((2, 3), (0, 0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title("people who are survived")
plt.ylabel("number of people")

plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel("number")
plt.xlabel("kind")
plt.title("ranking of people")

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel("age")
plt.grid(b=True, which='major',axis='y')
plt.title("possibility of survive counting on age")
plt.savefig("pic1.jpg")
# plt.subplot2grid((2, 3), (1, 0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel("age")
# plt.ylabel("midu")
# plt.title("age & ranking")
# plt.legend(('tou', '2deng', '3deng'), loc='best')
# #
survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救': survived_1, u'没获救': survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"乘客等级与获救情况关系")
plt.xlabel(u"等级")
plt.ylabel(u"人数")
plt.savefig('pic2.jpg')

survived_man = data_train.Survived[data_train.Sex == 'male'].value_counts()
survived_woman=data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({u'男性':survived_man , u'女性':survived_woman})
df.plot(kind='bar',stacked=True)
plt.title(u"性别与获救情况")
plt.xlabel(u"性别")
plt.ylabel(u"人数")
plt.savefig("pic3.jpg")
plt.show()
print('\n')
print(data_train.Cabin.value_counts())
