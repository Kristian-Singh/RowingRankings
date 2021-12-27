
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns



df=pd.DataFrame()
for p in Path('/Users/kristian/Desktop/rowdata/rawdata').glob('*.csv'):
    print(p)

    try:
        temp=pd.read_csv(p)
        temp.columns = temp.columns.str.lower()
        temp=temp[['raceid', 'date', 'crewtype', 'distance', 'team1', 'time1', 'team2','time2', 'team3', 'time3', 'team4', 'time4', 'team5', 'time5', 'team6','time6', 'team7', 'time7', 'team8', 'time8', 'racetype', 'racename']]
        temp['file']=str(p).split('/')[-1]
        print(temp)
        df=df.append(temp)
    except:
        fail.append(p)

df['date']=pd.to_datetime(df.date)
df=df.sort_values(by='date')
df=df.reset_index()
df=df.drop(axis=1,labels='index')
df['teams']=''


for i in range(len(df)):
    l=[]
    l.append(df.iloc[i].team1)
    l.append(df.iloc[i].team2)
    l.append(df.iloc[i].team3)
    l.append(df.iloc[i].team4)
    l.append(df.iloc[i].team5)
    l.append(df.iloc[i].team6)
    l.append(df.iloc[i].team7)
    l.append(df.iloc[i].team8)
    l = [x for x in l if type(x)== str]
    df.at[i,'teams']=l

df.raceid=df.index

df=pd.wide_to_long(df, stubnames=['team','time'], i=['raceid','date', 'crewtype','racetype','file'], j='number').reset_index()
df=df.reset_index()
df=df.drop(axis=1,labels='index')

df.to_csv('/Users/kristian/Desktop/rowdata/cleaned_data/all.csv')

#
