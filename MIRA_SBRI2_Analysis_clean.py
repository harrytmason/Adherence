# -*- coding: utf-8 -*-
"""

@author: Harry T. Mason

Load and read SBRI data extraction 04.2018v3 data from Dr Emma Stanmore

Linked to following paper:
Development and exploratory analysis of a multi-dimensional metric of adherence for physiotherapy e-Health interventions
Harry T Mason, Siobhán O’Connor, Emma Stanmore, David C Wong

This script will create the plots used in the figure, as well as other visualisations and analysis that were not included in the paper
"""
#%% Load libraries
import datetime as dt
import itertools
import matplotlib.dates as pltdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import plotly.express as px
import scipy.stats as spyst
from   sklearn.cluster import KMeans
from   sklearn.decomposition import PCA

#%% Load Dataset

#% Load dataset
dfc = pd.read_excel('G:/My Drive/MIRA/SBRI2 data extraction 04.2018v3/AllPatients.xlsx', engine='openpyxl')


#% Sanitise Datetime inputs
dfc['Timestamp']=pd.to_datetime(dfc['Timestamp'], infer_datetime_format=True)
dfc['Date']=pd.to_datetime(dfc['Date'], infer_datetime_format=True)

#Remove times where we only care about the Date
dfc['Date'] = [x.date() for x in dfc['Date']]

#% Removing Erroneous inputs (the NaNs not assigned to a patient)
dfc = dfc.loc[dfc["Name"].notnull()]

#%% Print Summary stats
dfc["Count"]=1

print("There are %d unique participants." % (dfc["Name"].unique().size))

dfSess = dfc.drop_duplicates(subset="SessionID").groupby("Name")["SessionID"].describe()['count']
dfSessDays = dfc.groupby(["Name"])["Date"].describe()['unique']

print("There are %d sessions, with %d-%d per patient (mean %2.1f)" % (dfc["SessionID"].unique().size, dfSess.min(), dfSess.max(), dfSess.mean()))
print("Using the definition of >3 sessions is compliance, %d participants achieved compliance" % sum(dfSess>3))
print("Naively assuming each patient was assigned 3 sessions for 12 weeks, %d participants completed >=36 sessions" % sum(dfSess>=36))
print("Relatedly, %d participants completed >=36 unique Session Days" %(sum(dfSessDays>=36)))
print("There are %d games (%d unique), assuming each game has one instance of 'Time' recording" % (dfc.loc[dfc["StatisticsName"]=="Time"].describe()["Count"][0],dfc["Game"].unique().size))

#%% Print Adherence as initial adoption

adopt=np.zeros(22)
adopt_days=np.zeros(22)
for x in np.arange(0,22):
    adopt[x]=sum(dfSess>x)
    adopt_days[x]=sum(dfSessDays>x)

fig0 = plt.figure(figsize=(10,7))
a0 = fig0.add_subplot(2,1,1)
plt.plot(np.arange(0,22),adopt)
plt.title('Adherence as Initial Adoption', fontsize=24)
plt.xlabel('Number of sessions', fontsize=12)
plt.ylabel('Participants still active\nafter X sessions', fontsize=12)
ya0max=a0.get_ylim()[1]
plt.plot([3.5, 3.5],[0, ya0max],'r--',linewidth=0.5)
plt.xticks(ticks=[0,3,6,9,12,15,18,21])
plt.grid(axis='x')
plt.ylim([40.5,56.5])
plt.xlim([0,21])

a1 = fig0.add_subplot(2,1,2)
plt.plot(np.arange(0,22),adopt_days)
a1.set_xlabel('Number of unique session days', fontsize=12)
a1.set_ylabel('Participants still active\nafter X sessions days', fontsize=12)
ya1max=a1.get_ylim()[1]
plt.plot([3.5, 3.5],[0, ya1max],'r--',linewidth=0.5)
plt.xticks(ticks=[0,3,6,9,12,15,18,21])
plt.grid(axis='x')
plt.ylim([40.5,56.5])
plt.xlim([0,21])



#%% Print Summary Graphs


#Sessions per patient histogram
fig2 = plt.figure(figsize=(10,7))
a=dfc.drop_duplicates("SessionID").groupby(["Name"])["SessionID"].describe()['count']
ax2=a.hist(bins=int(a.max()-a.min()+1))
y2max=ax2.get_ylim()[1]
ax2.plot([35.5, 35.5],[0, y2max],'k--',linewidth=0.5)
ax2.set_ylim([0,y2max])
ax2.set_xlabel('Number of Sessions', fontsize=16)
ax2.set_title('Distribution of #sessions by patient',fontsize=24)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.show()

#Games per session
fig3 = plt.figure(figsize=(10,7))
a=dfc.loc[dfc["StatisticsName"]=="Time"].groupby("SessionID")["StatisticsName"].describe()['count']      #Always cut down the dataset before describe() if possible
ax3=a.hist(bins=int(a.max()-a.min())*5+1)
ax3.set_xlabel('Number of Games', fontsize=16)
ax3.set_title('Distribution of #games by session',fontsize=24)
plt.show()

#Unique Session Dates per patient 
fig4 = plt.figure(figsize=(10,7))
a=dfc.groupby(["Name"])["Date"].describe()['unique']
ax4=a.hist(bins=int(a.max()-a.min()+1))
ax4.set_xlabel('Number of Sessions Days', fontsize=16)
ax4.set_title('Distribution of #sessions days by patient',fontsize=24)
ax4.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.show()


#%% Plot unique session days from start
 
b=dfc.groupby(["Name"])["Date"].describe()['unique']>19

zero_date=dt.datetime(2000,1,1)
zero_num = pltdates.date2num(zero_date)
f = plt.figure(figsize=(10,7))
ax = f.add_subplot(1,1,1)
for x in dfc["Name"].unique():
    if b[x]:
        c=dfc.loc[dfc["Name"]==x]["Date"].unique()-dfc.loc[dfc["Name"]==x]["Date"].unique().min()
        
        # convert datetimes to numbers
        time = [zero_date + t for t in c]
        time = [t-zero_num for t in pltdates.date2num(time)]
        time.sort()
        
        # plot
        y = np.ones(c.size)-[i for i in range(0,c.size)]/(np.ones(c.size)*(c.size-1))
        ax.plot(time,y, '-*')
        
ax.set_ylabel("Proportion of Session Days completed", fontsize=16)
ax.set_xlabel("Days since first session", fontsize=16) 
ax.set_title("Some participant session completion curves",fontsize=24)  
ax.grid()   
plt.show()


#%% Adherence as Dropout

b=dfc.groupby(["Name"])["Date"].describe()['unique']>0
i1 = 0
d = np.ones(sum(b))
zero_date=dt.datetime(2000,1,1)
zero_num = pltdates.date2num(zero_date)

for x in dfc["Name"].unique():  #could also use df.drop_duplicates(subset="PatientID")["PatientID"]
    if b[x]:
        tmp=(dfc.loc[dfc["Name"]==x]["Date"].unique()-dfc.loc[dfc["Name"]==x]["Date"].unique().min()).max()
        d[i1]=pltdates.date2num(zero_date+tmp)-zero_num
        i1 +=1

d.sort()
y = np.ones(d.size)-[i for i in range(0,d.size)]/(np.ones(d.size)*(d.size-1))
d = np.insert(d,0,0,axis=0)
y = np.insert(y,0,1,axis=0)  



fig=plt.figure(figsize=(8,8))
a1 = fig.add_subplot(2,1,1)

plt.plot(d,y)
a1.set_xlim([d.min(), d.max()])
a1.set_ylim([0,1])
a1.set_title('Adherence as Dropout', fontsize=24)
a1.set_xlabel("Days since first session", fontsize=12)     
a1.set_ylabel("Dropout Probability", fontsize=12)    
a1.grid(which='both')

b=dfc.drop_duplicates("SessionID").groupby(["Name"])["SessionID"].count()
b=b/36
b=b.sort_values()
dropOut = b
dropOut[dropOut>1]=1
y = 1-[i for i in range(0,b.size)]/(np.ones(b.size)*(b.size-1))

a2 = fig.add_subplot(2,1,2)
plt.plot(b*100,y)
a2.set_xlim([0, 100])
a2.set_ylim([0,1])
a2.set_xlabel("% of total prescribed sessions completed", fontsize=12)     
a2.set_ylabel("Dropout Probability", fontsize=12)        
a2.grid(which='both')
plt.show()

print('AUC of %%completed graph is %1.3f'%np.trapz(y[:np.where(b>=1)[0][0]],b[:np.where(b>=1)[0][0]]))

#%% Adherence as Consistency


zero_date=dt.datetime(2000,1,1)
zero_num = pltdates.date2num(zero_date)

for x in dfc["Name"].unique():  #could also use df.drop_duplicates(subset="PatientID")["PatientID"]
    dfc.loc[dfc["Name"]==x, "DayCount"] = dfc.loc[dfc["Name"]==x]["Date"]-dfc.loc[dfc["Name"]==x]["Date"].min()

dfc["WeekCount"] = np.floor((pltdates.date2num(zero_date+dfc["DayCount"])-zero_num)/7)

tmp=dfc.drop_duplicates("SessionID").groupby(["Name","WeekCount"])["Count"].describe()['count']>=3 #Drop any sessions beyond the 3rd within a week
weekCount=tmp.groupby("Name").sum().sort_values()
sum(weekCount>=12)

fig=plt.figure(figsize=(8,6))
a1 = fig.add_subplot(2,1,1)
a1.hist(weekCount,bins=np.arange(0,23)-0.475,rwidth=0.25)
y1max = a1.get_ylim()[1]
plt.plot([11.5, 11.5],[0, y1max],'r-',linewidth=0.5)
plt.xticks(ticks=[0,3,6,9,12,15,18,21])
plt.xlim([-0.2,21.2])
plt.grid()
a1.set_ylim([0, y1max])
a1.set_title('Adherence as Consistency', fontsize=24)
a1.set_ylabel('Number of Participants', fontsize=12)

count, bins_count = np.histogram(weekCount,bins=np.arange(0,23)-0.475)
pdf = count/sum(count)
cdf = np.cumsum(pdf)*np.sum(count)
a2 = fig.add_subplot(2,1,2)
plt.bar(np.arange(0,22),cdf,width=0.25)
plt.ylim(0,56)
y2max = a2.get_ylim()[1]
plt.plot([11.5, 11.5],[0, y2max],'r-',linewidth=0.5)
plt.xticks(ticks=[0,3,6,9,12,15,18,21])
plt.yticks(ticks=[0,8,16,24,32,40,48,56])
plt.xlim([-0.2,21.2])
plt.grid()
a2.set_ylim([0, y2max])
a2.set_xlabel('Number of Completed Weeks', fontsize=12)
a2.set_ylabel('Cumulative Count\nof "Finished" Participants', fontsize=12)
plt.show()

print("%d participants completed >=3 unique session days a week for 12 weeks" % sum(weekCount>11.5))


#%% Exercise per week

EPW=dfc.loc[dfc['StatisticsName']=="Time"].groupby(["Name","WeekCount"])["StatisticValue"].describe()

EPW['sum'] = EPW['count']*EPW['mean']


#%% Adherence as duration, Big comparison plot between 20 and 30 minute thresholds

# 20 minute plots

X=20
EPW['sumX'] = (EPW['sum']>(X*60))

EPWeekCount=EPW.groupby("Name").sum()

sum(EPWeekCount['sumX']>11.5)

fig=plt.figure(figsize=(16,8))
a1 = fig.add_subplot(2,2,1)
a1.hist(EPWeekCount['sumX'],bins=np.arange(0,18)-0.475,rwidth=0.25)
y1max = a1.get_ylim()[1]
plt.grid()
a1.set_xlim([-0.2,16.2])
plt.suptitle('Adherence as Duration of Exercise', fontsize=36,y=0.95)
a1.set_ylabel('Number of Participants', fontsize=16)

print("%d participants completed >=%d minutes of exercise a week for 12 weeks" % (sum(EPWeekCount['sumX']>11.5),X))

count, bins_count = np.histogram(EPWeekCount['sumX'],bins=np.arange(0,18)-0.475)
pdf = count/sum(count)
cdf = np.cumsum(pdf)*np.sum(count)
a2 = fig.add_subplot(2,2,3)
plt.bar(np.arange(0,17),cdf,width=0.25)
plt.ylim(0,56)
y2max = a2.get_ylim()[1]
plt.plot([11.5, 11.5],[0, y2max],'r-',linewidth=0.5)
plt.yticks(ticks=[0,8,16,24,32,40,48,56])
plt.xlim([-0.2,16.2])
plt.grid()
a2.set_ylim([0, y2max])
a2.set_xlabel('Number of weeks with >=%d mins of exercise'%X, fontsize=16)
a2.set_ylabel('Cumulative Count\nof "Finished" Participants', fontsize=16)

# 30 minute plots

Y=30
EPW['sumY'] = (EPW['sum']>(Y*60))

EPWeekCount=EPW.groupby("Name").sum()

sum(EPWeekCount['sumY']>11.5)

a3 = fig.add_subplot(2,2,2)
a3.hist(EPWeekCount['sumY'],bins=np.arange(0,18)-0.475,rwidth=0.25)
y3max = a3.get_ylim()[1]
a3.plot([11.5, 11.5],[0, np.max([y1max,y3max])],'r-',linewidth=0.5)
a3.set_xlim([-0.2,16.2])
a3.set_ylim([0, np.max([y1max,y3max])])
a1.plot([11.5, 11.5],[0, np.max([y1max,y3max])],'r-',linewidth=0.5)
a1.set_ylim([0, np.max([y1max,y3max])])
plt.grid()
print("%d participants completed >=%d minutes of exercise a week for 12 weeks" % (sum(EPWeekCount['sumY']>11.5),X))

count, bins_count = np.histogram(EPWeekCount['sumY'],bins=np.arange(0,18)-0.475)
pdf = count/sum(count)
cdf = np.cumsum(pdf)*np.sum(count)
a4 = fig.add_subplot(2,2,4)
plt.bar(np.arange(0,17),cdf,width=0.25)
plt.ylim(0,56)
y4max = a4.get_ylim()[1]
plt.plot([11.5, 11.5],[0, y4max],'r-',linewidth=0.5)
plt.yticks(ticks=[0,8,16,24,32,40,48,56])
plt.xlim([-0.2,16.2])
plt.grid()
a4.set_ylim([0, y4max])
a4.set_xlabel('Number of weeks with >=%d mins of exercise'%Y, fontsize=16)
fig.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

#%% k-means clustering for duration vs consistency

#%  k-means clustering for duration (20 mins) vs consistency

# create kmeans object
kmeans_X = KMeans(n_clusters=3)

# fit kmeans object to data
points_X=np.zeros([weekCount.size,2])
points_X[:,0] = np.array(weekCount.sort_index())
points_X[:,1] = np.array(EPWeekCount['sumX'].sort_index())
kmeans_X.fit(points_X)

# save new clusters for chart
y_km_X = kmeans_X.fit_predict(points_X)

# linear regression 
slope_X, intercept_X, r, p, std_err = spyst.linregress(points_X[:,1],points_X[:,0])

#%  k-means clustering for duration (30 mins) vs consistency

# create kmeans object
kmeans_Y = KMeans(n_clusters=4)

# fit kmeans object to data
points_Y=np.zeros([weekCount.size,2])
points_Y[:,0] = np.array(weekCount.sort_index())
points_Y[:,1] = np.array(EPWeekCount['sumY'].sort_index())

kmeans_Y.fit(points_Y)

# save new clusters for chart
y_km_Y = kmeans_Y.fit_predict(points_Y)

# linear regression 
slope_Y, intercept_Y, r, p, std_err = spyst.linregress(points_Y[:,1],points_Y[:,0])

# plot cycle
m_list = itertools.cycle(['*','o','v'])
c_list = itertools.cycle(["r","b","g"])

#% big plots

fig=plt.figure(figsize=(23/3*2-3,17/3))
plt.suptitle('Consistency vs Duration of Exercise',fontsize=32)
a1=plt.subplot(1,2,1)
for x in np.arange(np.max(y_km_X)+1):
    ind_x = np.where(y_km_X==x)[0]
    plt.scatter(points_X[ind_x,1], points_X[ind_x,0], s=20, c=next(c_list),marker=next(m_list))
plt.plot(np.arange(100)/5,slope_X*np.arange(100)/5+intercept_X,'k--', linewidth=1)
plt.xlim(-0.5,16.5)
plt.ylim(-0.5,21.5)
plt.yticks(ticks=[0,3,6,9,12,15,18,21])
plt.ylabel('Number of weeks with >=3 sessions', fontsize=14)
plt.xlabel('Number of weeks with >=%d mins of exercise'%X, fontsize=14)

a2=plt.subplot(1,2,2,sharey=a1)
for x in np.arange(np.max(y_km_Y)+1):
    ind_x = np.where(y_km_Y==x)[0]
    plt.scatter(points_Y[ind_x,1], points_Y[ind_x,0], s=20, c=next(c_list),marker=next(m_list))
plt.plot(np.arange(100)/5,slope_Y*np.arange(100)/5+intercept_Y,'k--', linewidth=1)
plt.yticks(ticks=[0,3,6,9,12,15,18,21])
plt.xlim(-0.5,16.5)
plt.ylim(-0.5,21.5)
plt.xlabel('Number of weeks with >=%d mins of exercise'%Y, fontsize=14)
fig.subplots_adjust(wspace=0.1)


#%% PCA - setting up the Matrix, doing full cross-scatter 


points_Y=np.zeros([weekCount.size,5])
weekCountY               =weekCount.sort_index()
weekCountY[weekCountY>12]=12
ExPerWeek20            =EPWeekCount['sumX'].sort_index()
ExPerWeek20[ExPerWeek20>12]=12
ExPerWeek30            =EPWeekCount['sumY'].sort_index()
ExPerWeek30[ExPerWeek30>12]=12
points_Y[:,0] = np.array(weekCountY/12)
points_Y[:,1] = np.array(ExPerWeek20/12)
points_Y[:,2] = np.array(ExPerWeek30/12)
points_Y[:,3] = np.array(dropOut.sort_index())
points_Y[:,4] = np.array(dfSess>6)
df_Y = pd.DataFrame(points_Y,columns=['Consistency','Duration (20min)', 'Duration (30min)','Dropout','Adoption'])

fig=pd.plotting.scatter_matrix(df_Y,alpha=0.5,diagonal='hist',hist_kwds={'bins':13},range_padding=0.2,figsize=(9,9))

plt.suptitle('Full Adherence Metric Scatter Matrix',fontsize=20,y=0.92)

#%% PCA - doing the PCA
pca = PCA(n_components=5)
tmp=pca.fit(points_Y)
print('Component contribution breakdown to a combined PCA:')
print(tmp.components_)
points_PCA = np.matmul(points_Y,tmp.components_)
df_PCA = pd.DataFrame(points_PCA,columns=['Component 1','Component 2', 'Component 3','Component 4','Component 5'])


fig=pd.plotting.scatter_matrix(df_PCA,alpha=0.5,diagonal='hist',hist_kwds={'bins':13},range_padding=0.2,figsize=(9,9))
plt.suptitle('Full Adherence Metric PCA Scatter Matrix',fontsize=20,y=0.92)
fig[0][0].set_yticklabels([0,0.2,0.4,0.6])


#%% Plot scatter matrix with principle adherence component

points_Z=np.zeros([weekCount.size,6])

points_Z[:,0] = np.array(weekCountY/12)
points_Z[:,1] = np.array(ExPerWeek20/12)
points_Z[:,2] = np.array(ExPerWeek30/12)
points_Z[:,3] = np.array(dropOut.sort_index())
points_Z[:,4] = np.array(dfSess>6)
points_Z[:,5] = np.array(df_PCA['Component 1'])
df_Z = pd.DataFrame(points_Z,columns=['Consistency','Duration (20min)', 'Duration (30min)','Dropout','Adoption','Combined PCA'])

fig=pd.plotting.scatter_matrix(df_Z,alpha=0.5,diagonal='hist',hist_kwds={'bins':13},range_padding=0.2,figsize=(10,10))
plt.suptitle('Full Adherence Metric Scatter Matrix',fontsize=20,y=0.92)

#%% Visualising individual data sessions (with options provided)
x='1J7finished 30.9.16'
x='1D2 017Finished 12th Aug'
x='2E1Dougrie'
x='1A10'
x='2A4'
Nsessions = dfc.loc[dfc["Name"]==x]["SessionID"].unique().size

fig = plt.figure(figsize=(10,7))
ax=fig.add_subplot(1,1,1)
ax.plot(range(1,Nsessions+1),dfc.loc[dfc["Name"]==x].drop_duplicates("SessionID")["Date"].sort_values(),'-x')
ax.set_xlabel("Session Number",fontsize=12)
ax.set_ylabel('Date of Session',fontsize=12)
ax.set_title(x,fontsize=24)
ax.yaxis.set_ticks(pd.date_range(dfc.loc[dfc["Name"]==x]["Date"].unique().min(), dfc.loc[dfc["Name"]==x]["Date"].unique().max()+dt.timedelta(6),freq='7D').tolist())

ax.xaxis.set_major_locator(ticker.MultipleLocator(3))

ax.set_xlim([0, Nsessions+1])
ax.grid(axis="both")

#%% Visualising individual data exercise time
x='1J7finished 30.9.16'
x='1D2 017Finished 12th Aug'
x='2E1Dougrie'
x='1A10'
x='2A4'
Weeks = dfc.loc[dfc["Name"]==x]["WeekCount"].unique()
EPW1=dfc.loc[dfc['Name']==x].loc[dfc['StatisticsName']=="Time"].groupby(["WeekCount"])["StatisticValue"].describe()
EPW1=EPW1['count']*EPW1['mean']
fig = plt.figure(figsize=(10,7))
ax=fig.add_subplot(1,1,1)
ax.plot(EPW1/60,'-x')
ax.set_xlabel("Week",fontsize=12)
ax.set_ylabel('Exercise Time per Week (minutes)',fontsize=12)
ax.set_title(x,fontsize=24)

# ax.set_xlim([0, Nsessions+1])
ax.grid()

#%% Plot 4 participants as a mega graph


S_sub = ['2A4','2E1Dougrie','1J7finished 30.9.16','1A10']
Nsub = len(S_sub)
fig = plt.figure(figsize=(Nsub*4,8))
fig.suptitle('Individual Exercise Curves',fontsize=32)
for x in np.arange(Nsub):
    
    
    Sx = S_sub[x]
    Nsessions = dfc.loc[dfc["Name"]==Sx]["SessionID"].unique().size
    Weeks = dfc.loc[dfc["Name"]==Sx]["WeekCount"].unique()
    EPW1=dfc.loc[dfc['Name']==Sx].loc[dfc['StatisticsName']=="Time"].groupby(["WeekCount"])["StatisticValue"].describe()
    EPW1=EPW1['count']*EPW1['mean']
    
    ax=fig.add_subplot(2,Nsub,x+1)
    ax.plot([dfc.loc[dfc["Name"]==Sx]["Date"].unique().min()+dt.timedelta(12*7),dfc.loc[dfc["Name"]==Sx]["Date"].unique().min()+dt.timedelta(12*7)],[0,40],'r',linewidth=2)
    ax.plot(dfc.loc[dfc["Name"]==Sx].drop_duplicates("SessionID")["Date"].sort_values(),range(1,Nsessions+1),'-x')

    if not x: ax.set_ylabel("Session Number", fontsize=16)
    ax.set_title("Participant %d"%(x+1), fontsize=22)
    
    plt.xticks(pd.date_range(dfc.loc[dfc["Name"]==Sx]["Date"].unique().min(), dfc.loc[dfc["Name"]==Sx]["Date"].unique().min()+dt.timedelta(7*21),freq='7D').tolist(),labels=[int(x) for x in np.arange(0,22*7,7)/7])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(3))
    ax.set_xlim([dfc.loc[dfc["Name"]==Sx]["Date"].unique().min()-dt.timedelta(1),dfc.loc[dfc["Name"]==Sx]["Date"].unique().min()+dt.timedelta(7*14)])
    ax.set_ylim([0,37])
    ax.grid()
    
    ax=fig.add_subplot(2,Nsub,x+1+Nsub)
    ax.plot([12,12],[0,100],'r',linewidth=2)

    ax.plot(np.arange(0,len(EPW1))+0.5,EPW1/60,'-x')
    if not x: ax.set_ylabel('Exercise Time per Week\n(minutes)', fontsize=16)
    
    plt.xticks([int(x) for x in np.arange(0,22*7,7)/7])
    ax.set_xlim([0-1/7,14])
    ax.set_ylim([0,75])
    ax.set_xlabel("Weeks Since Start"%x, fontsize=16)
    ax.grid()
    fig.subplots_adjust(hspace=0.1)

#%% End
