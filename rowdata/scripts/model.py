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

class RowModel:
    def __init__(self,path):
        with open(Path(path)) as json_file:
            self.team_info = json.load(json_file)

        for key in self.team_info.keys():
            self.team_info[key]=[np.array(self.team_info[key][0]),np.array(self.team_info[key][1])]

    def check_key(self,key):
        check=key in self.team_info.keys()
        return(check)

    def get_mean_estimates(self,key):
        key=key.upper()
        check=self.check_key(key)
        if check:
            sd=self.team_info[key][1].mean()
            mean=self.team_info[key][0].mean()
            return([mean,sd])
        else:
            raise Exception('Team Does Not Exist In Model!')

    def plot_info(self,key):
        key=key.upper()
        check=self.check_key(key)
        if check:
            fig, axs = plt.subplots(ncols=3)

            sns.scatterplot(x=self.team_info[key][0],y=self.team_info[key][1],ax=axs[0],s=5,color='black').axvline(self.get_mean_estimates(key)[0],color="red",linestyle='--')
            sns.lineplot(x=range(0,2000),y=self.team_info[key][0],ax=axs[1],color='#26abff').axhline(self.get_mean_estimates(key)[0],color="white",linestyle='--')
            sns.lineplot(x=range(0,2000),y=self.team_info[key][1],color='#FD5602',ax=axs[2]).axhline(self.get_mean_estimates(key)[1],color='white',linestyle='--')
            axs[0].axhline(self.get_mean_estimates(key)[1],color="red",linestyle='--')
            axs[0].set_title('Gibbs Sampler Region',fontsize=8)
            axs[0].set_ylabel('SD')
            axs[0].set_xlabel('Mu')
            axs[1].set_title('Trace for Mu',fontsize=8)
            axs[1].set_xlabel('Iteration')
            axs[2].set_title('Trace for SD',fontsize=8)
            axs[2].set_xlabel('Iteration')

            axs[0].set_xticklabels([str(i) for i in axs[0].get_xticks()], fontsize = 8)
            axs[1].set_xticklabels([str(i) for i in axs[1].get_xticks()], fontsize = 8)
            axs[2].set_xticklabels([str(i) for i in axs[2].get_xticks()], fontsize = 8)
            axs[0].set_yticklabels([str(i) for i in axs[0].get_yticks()], fontsize = 8)
            axs[1].set_yticklabels([str(i) for i in axs[1].get_yticks()], fontsize = 8)
            axs[2].set_yticklabels([str(i) for i in axs[2].get_yticks()], fontsize = 8)
            fig.tight_layout(pad=.35,rect=[0, 0.03, 1, 0.95])
            fig.suptitle(f"Scatter and Trace Plots for {key}\n\n", fontsize=10)

        else:
            raise Exception('Team Does Not Exist In Model!')

    def generate_probs(self,team1,team2,n_sim=10000,epsilon=.1):
        team1=self.get_mean_estimates(team1.upper())
        team2=self.get_mean_estimates(team2.upper())

        race_results=[]
        team1_time=sps.norm.rvs(team1[0],team1[1],size=n_sim)
        team2_time=sps.norm.rvs(team2[0],team2[1],size=n_sim)

        for i in range(n_sim):
            dist_between_boats=abs(team1_time[i]-team2_time[i])
            if (dist_between_boats==0 or dist_between_boats<=epsilon):
                race_results.append('tie')
            if (team1_time[i]<team2_time[i] and dist_between_boats>=epsilon ):
                race_results.append('team1')
            if (team1_time[i]>team2_time[i] and dist_between_boats>=epsilon):
                race_results.append('team2')

        team1_prob=np.count_nonzero(np.array(race_results) == 'team1')/n_sim
        team2_prob=np.count_nonzero(np.array(race_results) == 'team2')/n_sim
        tie_prob=np.count_nonzero(np.array(race_results) == 'tie')/n_sim

        return([team1_prob,team2_prob,tie_prob])

    def get_head_probs(self,team1,team2,runs=10000,epsilon=.1):
        team1_prob=[]
        team2_prob=[]
        tie_prob=[]
        check1=self.check_key(team1.upper())
        check2=self.check_key(team1.upper())
        if check1 and check2:
            for run in range(runs):
                probs=self.generate_probs(team1,team2,epsilon=epsilon)
                # print(f"Run: {run}; Probability {team1} wins :{probs[0]}; Probability {team2} wins :{probs[1]} ; Probability 'tie' wins :{probs[2]}")
                team1_prob.append(probs[0])
                team2_prob.append(probs[1])
                tie_prob.append(probs[2])

            #Plotting Probabilities
            fig, axs = plt.subplots(nrows=3)

            sns.lineplot(data=np.array(team1_prob[::10]),ax=axs[0]).axhline(np.array(team1_prob).mean(),color='#FD5602',linestyle='--',linewidth=.75)
            sns.lineplot(data=np.array(team2_prob[::10]),ax=axs[1]).axhline(np.array(team2_prob).mean(),color='#FD5602',linestyle='--',linewidth=.75)
            sns.lineplot(data=np.array(tie_prob[::10]),ax=axs[2]).axhline(np.array(tie_prob).mean(),color='#FD5602',linestyle='--',linewidth=.75)

            axs[0].set_title(f"Probability {team1.upper()} Wins",fontsize=8)
            # axs[0].set_ylabel('SD')
            # axs[0].set_xlabel('Mu')
            axs[1].set_title(f"Probability {team2.upper()} Wins",fontsize=8)
            # axs[1].set_xlabel('Iteration')
            axs[2].set_title('Probability of Tie',fontsize=8)
            # axs[2].set_xlabel('Iteration')

            axs[0].tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False)
            axs[1].tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False)
            axs[2].tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False)
            axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axs[0].set_yticklabels([str(i) for i in axs[0].get_yticks()], fontsize = 8)
            axs[1].set_yticklabels([str(i) for i in axs[1].get_yticks()], fontsize = 8)
            axs[2].set_yticklabels([str(i) for i in axs[2].get_yticks()], fontsize = 8)
            fig.tight_layout()
            # fig.suptitle(f"Win Probabilities For {team1} and {team2}\n\n", fontsize=10)

            return([team1_prob,team2_prob,tie_prob])
        else:
            raise Exception('Team Does Not Exist In Model!')

    def sim_race(self,teams:list,runs=10000,n_sim=1000):
        for team in teams:
            check=self.check_key(team.upper())
            if check==False:
                raise Exception('Team Does Not Exist In Model!')
            else:
                pass
        probs={}
        times={}
        for team in teams:
            probs[team]=[]
        for run in range(runs):
            for team in teams:
                times[team]=sps.norm.rvs(self.get_mean_estimates(team.upper())[0],self.get_mean_estimates(team.upper())[1],size=n_sim)
            for i in range(n_sim):
                result=[]
                for team in teams:
                    result.append(times[team][i])
                result.sort()
                for pos in range(0,len(result)):
                    for team in teams:
                        if result[pos]==times[team][i]:
                            times[team][i]=pos+1
                        else:
                            pass
            for team in teams:
                probs[team].append((np.unique(times[team], return_counts=True)[1]/n_sim))
        r=self.plot_race(teams=teams,p=probs,runs=runs)
        pred=self.rank_race(p=r,teams=teams)
        return({"probs":r,"predictions":pred})

    def plot_race(self,teams,p,runs):
        results=pd.DataFrame()
        slicer=int(round(int(np.sqrt(runs))))
        for team in teams:
            cols=[]
            for k in range(len(teams)):
                cols.append(str(k+1))
            temp=pd.DataFrame(p[team])
            temp=temp.iloc[::slicer, :]
            temp.columns=cols
            temp['iteration']=temp.index
            temp=pd.melt(temp, id_vars='iteration', value_vars=cols)
            temp.columns=['iteration', 'Result', 'prob']
            temp['team']=team
            results=results.append(temp)
        results.groupby(['team','Result']).prob.describe().reset_index()[['team','Result','mean','50%']].sort_values(ascending=False,by="mean")
        sns.relplot(data=results,x="iteration", y="prob",hue="Result", col="team",kind="line",aspect=.5)
        return(results)

    def rank_race(self,p,teams):
        pred=p.groupby(['team','Result']).prob.describe().reset_index()[['team','Result','mean']]
        predictions={}
        for i in range(len(teams)):
            pos=str(i+1)
            place=pred.query('Result==@pos').sort_values(ascending=False,by='mean').iloc[0].team
            pred=pred.query('team != @place')
            predictions[place]=pos
        return(predictions)






### TESTING THE CLASS FUNCTIONALITY BELOW
Model=RowModel("/Users/kristian/Desktop/rowdata/cleaned_data/team_models.json")

Model.get_mean_estimates('PRIN')

Model.plot_info('CORN')

Model.get_head_probs('harv','prin',runs=1000)

pred=Model.sim_race(teams=['CAL','YALE','BRWN','WASH'],runs=1000)








#
