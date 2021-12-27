import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
from pathlib import Path
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
%config InlineBackend.figure_format = 'retina'
df=pd.read_csv(Path('cleaned_data/all.csv'))
df=df.query('file!="RowData2012.csv"')


df=df[['raceid', 'date', 'crewtype', 'racetype', 'file',
       'number', 'teams', 'racename', 'distance', 'team', 'time', 'minutes',
       'seconds', 'time_in_secs']]
df['date']=pd.to_datetime(df['date'])
df['day']=df.date.dt.day
df['month']=df.date.dt.month
df['year']=df.date.dt.year
df=df[['raceid', 'date','day', 'month', 'year', 'crewtype', 'racetype', 'file', 'number', 'teams',
       'racename', 'distance', 'team', 'time', 'minutes', 'seconds',
       'time_in_secs']]


AverageTimes=pd.DataFrame()
 for dt in df.date.unique()[1:]:
     row=df.query('date<@dt & distance>0.0').groupby(['crewtype','distance']).time_in_secs.describe().reset_index()[['crewtype','distance','mean','50%']]
     row['upto_date']=dt
     AverageTimes=AverageTimes.append(row)

AverageTimes.columns=['crewtype', 'distance', 'mean_race_time', 'median_race_time', 'upto_date']

### MODELING AFFECT OF THE SEASON TIME ON RACE TIME, -->
### BOATS SEEM TO GET SLOWER EACH MONTH IN A SEASON
### BUT GOOD BOATS SLOW DOWN AT A RATE LESS THAN BAD BOATS

df.racetype.astype("str")
df.groupby('month').raceid.nunique()
df.groupby(['file','racetype']).raceid.nunique()

pseudo_skill=df.query("crewtype=='MHV' & distance==2000.0").groupby(['year','team'])\
.time_in_secs.describe().reset_index()[['team','year','mean']].sort_values(['team','year'])


reg_data = df.query("crewtype=='MHV' & distance==2000.0")[['team','year','month', 'racetype', 'time_in_secs']].dropna()
reg_data=reg_data.merge(pseudo_skill).sort_values(['team','year'])



mod=smf.ols(formula='time_in_secs ~ racetype + month*mean', data=reg_data)
res=mod.fit()
print(res.summary())
######################
###  Quick analysis on regressing average yearly time on previous 4 years of average times
###  Better would be to look at each race and do both lag of previous races in the season and average lags
###  This combines streakiness effects really good recent races or really shit recent races and long term effects i.e. yearly averages

times=pseudo_skill.pivot(index='team', columns='year', values='mean').reset_index().dropna()
times=times[times[2019].notna()]
times.columns=['team', 'y2011', 'y2013', 'y2014', 'y2015', 'y2016', 'y2017', 'y2018', 'y2019']

t1=times[['team','y2019','y2018','y2017','y2016','y2015']]
t1.columns=['team','y','lag1','lag2','lag3','lag4']
t2=times[['team','y2018','y2017','y2016','y2015','y2014']]
t2.columns=['team','y','lag1','lag2','lag3','lag4']
t3=times[['team','y2017','y2016','y2015','y2014','y2013']]
t3.columns=['team','y','lag1','lag2','lag3','lag4']


reg_time_df=t1.append(t2.append(t3))

mod_time_weights=smf.ols(formula='y~lag1+lag2+lag3+lag4', data=reg_time_df)
res_time_weights=mod_time_weights.fit()
print(res_time_weights.summary())

beta_params={}

beta_params['1']=res_time_weights.params[1]/res_time_weights.params[1:].sum()
beta_params['2']=res_time_weights.params[2]/res_time_weights.params[1:].sum()
beta_params['3']=res_time_weights.params[3]/res_time_weights.params[1:].sum()
beta_params['4']=res_time_weights.params[4]/res_time_weights.params[1:].sum()


######################################################################################################################################################
############# CREATING A LAGGED DATA FRAME SO WE CAN APPEND ON PREVIOUS RACE TIMES FOR EACH TEAM (Mens HV and 2000 meter races only!!) #################
#####################################################################################################################################################
races_per_team=df.query('distance==2000.0 & crewtype=="MHV"').groupby('team').raceid.unique().reset_index()

lag_df=pd.DataFrame(columns=['team','race','idlag1','idlag2','idlag3','idlag4','idlag5'])
for row in range(len(races_per_team)):
    team=races_per_team.at[row,'team']
    if len(races_per_team.at[row,'raceid'])>=5:
        for id in range(5,len(races_per_team.at[row,'raceid'])):
            current_race=races_per_team.at[row,'raceid'][id]
            race_lag1=races_per_team.at[row,'raceid'][id-1]
            race_lag2=races_per_team.at[row,'raceid'][id-2]
            race_lag3=races_per_team.at[row,'raceid'][id-3]
            race_lag4=races_per_team.at[row,'raceid'][id-4]
            race_lag5=races_per_team.at[row,'raceid'][id-5]
            temp=pd.DataFrame({
            "team":[team],
            'race':[current_race],
            "idlag1":[race_lag1],
            "idlag2":[race_lag2],
            "idlag3":[race_lag3],
            "idlag4":[race_lag4],
            "idlag5":[race_lag5]
            })
            lag_df=lag_df.append(temp)

    else:
        None

lag_df.to_csv('/Users/kristian/Desktop/rowdata/cleaned_data/lagged_races.csv')
######################
### CREATING THE BIG LAGGED DATA FRAME FOR REGRESSION MODEL
lag_df=pd.read_csv('cleaned_data/lagged_races.csv')
lag_df=lag_df[['team', 'race', 'idlag1', 'idlag2', 'idlag3', 'idlag4','idlag5']]

lag1=df.query('distance==2000.0 & crewtype=="MHV"')[['raceid','team','date','racetype','time_in_secs','teams','racename']]
lag1.columns=['idlag1','team','date_lag_1','racetype_lag_1','time_in_secs_lag_1','teams_lag_1','racename_lag_1']
lag2=df.query('distance==2000.0 & crewtype=="MHV"')[['raceid','team','date','racetype','time_in_secs','teams','racename']]
lag2.columns=['idlag2','team','date_lag_2','racetype_lag_2','time_in_secs_lag_2','teams_lag_2','racename_lag_2']
lag3=df.query('distance==2000.0 & crewtype=="MHV"')[['raceid','team','date','racetype','time_in_secs','teams','racename']]
lag3.columns=['idlag3','team','date_lag_3','racetype_lag_3','time_in_secs_lag_3','teams_lag_3','racename_lag_3']
lag4=df.query('distance==2000.0 & crewtype=="MHV"')[['raceid','team','date','racetype','time_in_secs','teams','racename']]
lag4.columns=['idlag4','team','date_lag_4','racetype_lag_4','time_in_secs_lag_4','teams_lag_4','racename_lag_4']
lag5=df.query('distance==2000.0 & crewtype=="MHV"')[['raceid','team','date','racetype','time_in_secs','teams','racename']]
lag5.columns=['idlag5','team','date_lag_5','racetype_lag_5','time_in_secs_lag_5','teams_lag_5','racename_lag_5']

lagged_mens_df=df.query('distance==2000.0 & crewtype=="MHV"').merge(lag_df,left_on=['team','raceid'],right_on=['team','race'])


lagged_mens_df=lagged_mens_df.merge(lag1).merge(lag2).merge(lag3).merge(lag4).merge(lag5)

yearly_average_times=df.query('distance==2000.0 & crewtype=="MHV"').groupby(['team',pd.to_datetime(df.date).dt.year]).time_in_secs.describe().reset_index()[['team','mean','date']]
yearly_average_times.columns=['team','average_time','race_year']


lagged_mens_df['race_year']=pd.to_datetime(lagged_mens_df['date']).dt.year
lagged_mens_df['race_year_lag1']=lagged_mens_df['race_year']-1
lagged_mens_df['race_year_lag2']=lagged_mens_df['race_year']-2
lagged_mens_df['race_year_lag3']=lagged_mens_df['race_year']-3
lagged_mens_df['race_year_lag4']=lagged_mens_df['race_year']-4


lagged_mens_df=lagged_mens_df.merge(yearly_average_times,left_on=['team','race_year_lag1'],right_on=['team','race_year'],suffixes=("","_lag1"))
lagged_mens_df=lagged_mens_df.merge(yearly_average_times,left_on=['team','race_year_lag2'],right_on=['team','race_year'],suffixes=("","_lag2"))
lagged_mens_df=lagged_mens_df.merge(yearly_average_times,left_on=['team','race_year_lag3'],right_on=['team','race_year'],suffixes=("","_lag3"))
lagged_mens_df=lagged_mens_df.merge(yearly_average_times,left_on=['team','race_year_lag4'],right_on=['team','race_year'],suffixes=("","_lag4"))

lagged_mens_df['date']=pd.to_datetime(lagged_mens_df['date'])
lagged_mens_df['teams_in_race']=lagged_mens_df['teams'].str.count(",")+1
lagged_mens_df['days_off']=(pd.to_datetime(lagged_mens_df['date'])-pd.to_datetime(lagged_mens_df['date_lag_1'])).dt.days
lagged_mens_df=lagged_mens_df.drop(axis=1,labels='Unnamed: 0')

lagged_mens_df['race_month']=lagged_mens_df.date.dt.month
lagged_mens_df['average_time_lag1']=lagged_mens_df['average_time']
lagged_mens_df.to_csv('/Users/kristian/Desktop/rowdata/cleaned_data/regression_data/lagged_regression_data.csv')




######################################################################################################################################################
################################################## More Clean up of Lagged Df #####################################################################
#####################################################################################################################################################
lagged_mens_df=pd.read_csv('/Users/kristian/Desktop/rowdata/cleaned_data/regression_data/lagged_regression_data.csv')

lagged_mens_df['date_lag_1']=pd.to_datetime(lagged_mens_df.date_lag_1)
lagged_mens_df['date_lag_2']=pd.to_datetime(lagged_mens_df.date_lag_2)
lagged_mens_df['date_lag_3']=pd.to_datetime(lagged_mens_df.date_lag_3)
lagged_mens_df['date_lag_4']=pd.to_datetime(lagged_mens_df.date_lag_4)
lagged_mens_df['date_lag_5']=pd.to_datetime(lagged_mens_df.date_lag_5)


AverageTimes.query('crewtype=="MHV" & distance==2000.0')

lagged_mens_df=lagged_mens_df.merge(AverageTimes.query('crewtype=="MHV" & distance==2000.0').\
add_suffix('_lag_1')[['mean_race_time_lag_1','median_race_time_lag_1','upto_date_lag_1']],right_on='upto_date_lag_1',left_on='date_lag_1').\
drop_duplicates()

lagged_mens_df=lagged_mens_df.merge(AverageTimes.query('crewtype=="MHV" & distance==2000.0').\
add_suffix('_lag_2')[['mean_race_time_lag_2','median_race_time_lag_2','upto_date_lag_2']],right_on='upto_date_lag_2',left_on='date_lag_2').\
drop_duplicates()

lagged_mens_df=lagged_mens_df.merge(AverageTimes.query('crewtype=="MHV" & distance==2000.0').\
add_suffix('_lag_3')[['mean_race_time_lag_3','median_race_time_lag_3','upto_date_lag_3']],right_on='upto_date_lag_3',left_on='date_lag_3').\
drop_duplicates()

lagged_mens_df=lagged_mens_df.merge(AverageTimes.query('crewtype=="MHV" & distance==2000.0').\
add_suffix('_lag_4')[['mean_race_time_lag_4','median_race_time_lag_4','upto_date_lag_4']],right_on='upto_date_lag_4',left_on='date_lag_4').\
drop_duplicates()

lagged_mens_df=lagged_mens_df.merge(AverageTimes.query('crewtype=="MHV" & distance==2000.0').\
add_suffix('_lag_5')[['mean_race_time_lag_5','median_race_time_lag_5','upto_date_lag_5']],right_on='upto_date_lag_5',left_on='date_lag_5').\
drop_duplicates()
lagged_mens_df.to_csv('/Users/kristian/Desktop/rowdata/cleaned_data/regression_data/lagged_regression_data.csv')




######################################################################################################################################################
################################################## Starting Regression Model #####################################################################
#####################################################################################################################################################


lagged_mens_df=pd.read_csv('/Users/kristian/Desktop/rowdata/cleaned_data/regression_data/lagged_regression_data.csv')


lagged_mens_df.columns

lagged_mens_df['date']=pd.to_datetime(lagged_mens_df['date'])
lagged_mens_reg_df=lagged_mens_df[['racetype','distance', 'team',
'time_in_secs', 'time_in_secs_lag_1','time_in_secs_lag_2', 'time_in_secs_lag_3', 'time_in_secs_lag_4',
'time_in_secs_lag_5','mean_race_time_lag_1','mean_race_time_lag_2','mean_race_time_lag_3','mean_race_time_lag_4','mean_race_time_lag_5',
'median_race_time_lag_1','median_race_time_lag_2','median_race_time_lag_3','median_race_time_lag_4','median_race_time_lag_5',
'average_time_lag1','average_time_lag2','average_time_lag3', 'average_time_lag4',
'teams_in_race', 'days_off', 'race_month','date']]


lagged_mens_reg_df=lagged_mens_reg_df.merge(AverageTimes.query('crewtype=="MHV" & distance==2000.0')[['mean_race_time','median_race_time','upto_date']],right_on='upto_date',left_on='date').\
drop_duplicates()


lagged_mens_reg_df['diff_from_avg']=lagged_mens_reg_df.time_in_secs-lagged_mens_reg_df.mean_race_time
lagged_mens_reg_df['diff_from_avg_lag_1']=lagged_mens_reg_df.time_in_secs_lag_1-lagged_mens_reg_df.mean_race_time_lag_1
lagged_mens_reg_df['diff_from_avg_lag_2']=lagged_mens_reg_df.time_in_secs_lag_2-lagged_mens_reg_df.mean_race_time_lag_2
lagged_mens_reg_df['diff_from_avg_lag_3']=lagged_mens_reg_df.time_in_secs_lag_3-lagged_mens_reg_df.mean_race_time_lag_3
lagged_mens_reg_df['diff_from_avg_lag_4']=lagged_mens_reg_df.time_in_secs_lag_4-lagged_mens_reg_df.mean_race_time_lag_4
lagged_mens_reg_df['diff_from_avg_lag_5']=lagged_mens_reg_df.time_in_secs_lag_5-lagged_mens_reg_df.mean_race_time_lag_5

from sklearn.preprocessing import StandardScaler

lagged_mens_reg_df[['diff_from_avg','diff_from_avg_lag_1','diff_from_avg_lag_2','diff_from_avg_lag_3','diff_from_avg_lag_4','diff_from_avg_lag_5']]

lagged_mens_reg_df[
['diff_from_avg','diff_from_avg_lag_1','diff_from_avg_lag_2',
'diff_from_avg_lag_3','diff_from_avg_lag_4','diff_from_avg_lag_5',
'time_in_secs', 'time_in_secs_lag_1','time_in_secs_lag_2', 'time_in_secs_lag_3', 'time_in_secs_lag_4',
'average_time_lag1','average_time_lag2','average_time_lag3', 'average_time_lag4','teams_in_race','days_off']]=StandardScaler().fit_transform(lagged_mens_reg_df[
['diff_from_avg','diff_from_avg_lag_1','diff_from_avg_lag_2',
'diff_from_avg_lag_3','diff_from_avg_lag_4','diff_from_avg_lag_5',
'time_in_secs', 'time_in_secs_lag_1','time_in_secs_lag_2', 'time_in_secs_lag_3', 'time_in_secs_lag_4',
'average_time_lag1','average_time_lag2','average_time_lag3', 'average_time_lag4','teams_in_race','days_off']])




model=smf.ols(formula='''time_in_secs~C(racetype)+time_in_secs_lag_1+time_in_secs_lag_2+ time_in_secs_lag_3+ time_in_secs_lag_4+
time_in_secs_lag_5+average_time_lag1+average_time_lag2+average_time_lag3+ average_time_lag4+
teams_in_race+ days_off+ C(race_month)''', data=lagged_mens_reg_df)
model=model.fit()
print(model.summary())









model=smf.ols(formula='''diff_from_avg~C(racetype)+diff_from_avg_lag_1+diff_from_avg_lag_2+ diff_from_avg_lag_3+ diff_from_avg_lag_4+
diff_from_avg_lag_5+average_time_lag1+average_time_lag2+average_time_lag3+ average_time_lag4+
teams_in_race+ days_off+ C(race_month)''', data=lagged_mens_reg_df)
model=model.fit()
print(model.summary())
















#
###### USEFUL FOR PLOTTING RELATIONSHIP BETWEEN LAGGED RACES AND CURRENT RACE

avg_lag=pd.concat([lagged_mens_df.groupby('team').time_in_secs.describe().reset_index()[['team','mean']],
lagged_mens_df.groupby('team').time_in_secs_lag_1.describe().reset_index()[['team','mean']]],axis=1)
avg_lag=pd.concat([avg_lag,lagged_mens_df.groupby('team').time_in_secs_lag_2.describe().reset_index()[['team','mean']]],axis=1)
avg_lag=pd.concat([avg_lag,lagged_mens_df.groupby('team').time_in_secs_lag_3.describe().reset_index()[['team','mean']]],axis=1)
avg_lag=pd.concat([avg_lag,lagged_mens_df.groupby('team').time_in_secs_lag_4.describe().reset_index()[['team','mean']]],axis=1)
avg_lag=pd.concat([avg_lag,lagged_mens_df.groupby('team').time_in_secs_lag_5.describe().reset_index()[['team','mean']]],axis=1)
avg_lag.columns=['team', 'mean_lag0', 'team1', 'mean_lag1', 'team2', 'mean_lag2', 'team3', 'mean_lag3', 'team4','mean_lag4', 'team5', 'mean_lag5']
avg_lag=avg_lag[['team', 'mean_lag0', 'mean_lag1','mean_lag2','mean_lag3','mean_lag4','mean_lag5']]
avg_lag=pd.wide_to_long(avg_lag, stubnames='mean_lag', i=['team'],j='lag').reset_index()


#
