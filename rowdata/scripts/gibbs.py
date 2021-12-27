##########################################################################
import pandas as pd
import numpy as np
from pathlib import Path
import scipy as sp
import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%config InlineBackend.figure_format = 'retina'
import json


df=pd.read_csv('/Users/kristian/Desktop/rowdata/cleaned_data/all.csv')

df_mens_varisty=df.query('crewtype=="MHV"')
mean_mens=df_mens_varisty.time_in_secs.describe()['50%']
#6:11.7


df_mens_varisty['diff_from_average']=df_mens_varisty['time_in_secs']-mean_mens
df_mens_varisty.diff_from_average.describe()

df_mens_varisty=df_mens_varisty.query('distance==2000.0')
df_mens_varisty.describe()

#
df_mens_varisty.query('team=="UCLA" | team=="WASH"| team=="CAL" | team=="YALE" ').sort_values(by='diff_from_average',ascending=False).hist(column='diff_from_average',by='team')
#
# df_mens_varisty.query('distance==2000 & team=="YALE"').sort_values(by='diff_from_average',ascending=False)
df_mens_varisty['date']=pd.to_datetime(df_mens_varisty['date'])
df_mens_varisty['month']=df_mens_varisty['date'].dt.month
df_mens_varisty.query('team=="CORN" | team=="WASH"| team=="CAL" | team=="YALE" ').groupby(['team','month']).diff_from_average.describe()
#
# df_mens_varisty.q




stats=df_mens_varisty[['team','diff_from_average']]\
.groupby('team').describe().reset_index().droplevel(0, axis=1) \
.sort_values(by="mean")

stats.columns=['team', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

def gibbs_sample_nig(X,mu0=0,sigma0=40,a0=4,b0=4,trials=11000):
    n=len(X)

    sd=[X.describe()['std']**2]
    mu=[]

    for trial in range(trials):
        sigma_kn=1/((1/sigma0)+n/sd[-1])
        mu_kn=sigma_kn*((mu0/sigma0)+(X.mean()*n)/sd[-1])
        mu.append(sps.norm.rvs(mu_kn,np.sqrt(sigma_kn)))

        sd.append(sps.invgamma.rvs(size=1,a=a0+(n/2),scale=b0+(((n-1)/2)*X.describe()['std']**2)+((n/2)*(X.mean()-mu[-1])**2))[0])
        # print(f"trial is: {trial}, mu value: {mu[-1]}, var value:{sd[-1]}")

    return(np.array(mu),np.array(sd))


#
# df_mens_varisty.query('team=="WASH"').hist(column='diff_from_average')
# x = np.linspace(mu - 3*sd.mean(), mu + 3*sd.mean(), 100)
# plt.plot(x, sps.norm.pdf(x, mu.mean(), sd.mean()))
# plt.show()



train=df_mens_varisty.query('file!="RowData2019.csv"')

stats=train[['team','diff_from_average']]\
.groupby('team').describe().reset_index().droplevel(0, axis=1) \
.sort_values(by="mean")

stats.columns=['team', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
teams=stats.query('count>10').team.unique()



# train.query('team=="UCLA" | team=="WASH"| team=="CORN" | team=="VILN" ').hist(column='diff_from_average',by='team')
# Generating Gibbs Sampler Estimates for Mu and Sd for each team with >= 10 obs
team_info={}
for team in teams:
    #  Filter worst and best record to avoid spurious results
    X=df_mens_varisty.query('team==@team').diff_from_average.sort_values()
    X=X[1:-1]
    # Send data to the Gibbs Sampler
    mu,var=gibbs_sample_nig(X)
    sd=np.sqrt(var)
    print(f"Team is: {team}; Mu estimated: {mu.mean()}; SD estimated: {sd.mean()}")

    mu=mu[999:-1]
    mu=mu[::5]
    sd=sd[1000:-1]
    sd=sd[::5]
    team_info[team]=[mu.tolist(),sd.tolist()]


a_file = open(Path("/Users/kristian/Desktop/rowdata/cleaned_data/team_models.json"), "w")
json.dump(team_info, a_file)
a_file.close()








#
