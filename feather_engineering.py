# from sklearn.ensemble import RandomForestRegressor
# import pandas as pd
# #数据拟合
# def set_missing_ages(df):
#     age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
#     known_age =age_df[age_df.Age.notnull()].as_matrix()
#     unknown_age=age_df[age_df.Age.isnull()].as_matrix()
#
#     y=known_age[:, 0]
#     x=known_age[:, 1:]
#     rfr=RandomForestRegressor(random_state=0, n_estimators=2000,n_jobs=-1)
#     rfr.fit(x,y)
#
#     prediction = rfr.predict(unknown_age[:,1::])
#     df.loc[(df.Age.isnull()),'Age']=prediction
#
#     return df, rfr
# def set_Cabin_type(df):
#     df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
#     df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
#     return df
# if __name__=='__main__':
#     data_train = pd.read_csv("S:/李帅的技能书/train.csv")
#     data_train,rfr=set_missing_ages(data_train)
#     data_train=set_missing_ages(data_train)
#     print(data_train)
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df
data_train = pd.read_csv("S:/李帅的技能书/train.csv")
data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
print(data_train)
print(data_train.info())
print(data_train.describe())