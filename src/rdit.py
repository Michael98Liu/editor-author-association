from statsmodels.formula.api import ols, logit
from statsmodels.api import Logit, OLS
from scipy.stats import ttest_ind, sem
from statsmodels.iolib.summary2 import summary_col
import scipy
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib

cm = 1/2.54  # centimeters in inches
plt.rcParams.update({'font.size': 7})

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams['grid.linewidth'] = 0.5
matplotlib.rcParams['axes.linewidth'] = 0.5

plt.rcParams["pdf.use14corefonts"] = True
plt.rcParams["font.family"] = "sans-serif"

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def CI(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

from datetime import datetime, timedelta

def converMonth(x):
    
    return (x.year-2000)*12 + x.month

def analysis(df, x, y, cutoff=141, RANGE=12, ax=None, markersize=3, integerY=True):
    
    xmin=cutoff-(RANGE)
    xmax=cutoff+(RANGE-1)
    
    if ax is None:
        ax = plt.gca()
        
    if integerY:
        df = df.assign(**{y: df[y].apply(int)})
    
    df = df[(df[x] >= xmin) & (df[x] <= xmax)]
    df = df.assign(After = lambda df: df[x] > cutoff) # cutoff belongs to "before", not "after"

    means = df.groupby(x)[y].mean().reset_index().sort_values(by=x)
    errors = df.groupby(x)[y].apply(CI).reset_index().sort_values(by=x)
    xvals = means[x]

    ax.errorbar(x=xvals, y=means[y],
            ls='none', marker='o', markersize=markersize, markerfacecolor='none')

    ax.plot([cutoff-0.5, cutoff-0.5], [0, 0.5], color='grey',lw=1,ls='--')

    a, b = np.polyfit(xvals[:RANGE], means[y][:RANGE], 1)
    ax.plot(xvals[:RANGE], a*np.array(xvals.values[:RANGE])+b, color='red')

    a, b = np.polyfit(xvals[RANGE:], means[y][RANGE:], 1)
    ax.plot(xvals[RANGE:], a*np.array(xvals.values[RANGE:])+b, color='red')

    
    df[x] = df[x].apply(lambda val: val-cutoff) # set cutoff to be 0
    # so that the coefficient of [After] shows the difference at the cutoff
        
    model = ols(formula=f'{y} ~ {x} + After + After:{x}', data=df.groupby([x,'After'])[y].mean().reset_index())

    results = model.fit(cov_type='HC3')
    
    return df, results

def plotRDiT(RANGE, COIoverTime, COIoverTimePLOS, mainTable=True):
    
    # if maintable set to true: show results related to the main effects (treatment)
    # if maintable set to false: show control effects
    
    ### axis and layout ###
    fig = plt.figure(layout='constrained', figsize=(18*cm, 10*cm))

    subfigs = fig.subfigures(2, 1, height_ratios=[5, 4])

    ax11, ax21, ax31 = subfigs[0].subplots(1, 3)
    ax12, ax22, ax32 = subfigs[1].subplots(1, 3)
    
    axes = [ax11, ax12, ax21, ax22, ax31, ax32]
    ### axis and layout ###
    
    
    ### values to be printed ###
    policyEst = []
    policyP = []
    timeEst = []
    timeP = []
    timePolicyEst = []
    timePolicyP = []
    N = []
    
    def addRegRes(regRes,policyEst=policyEst,policyP=policyP,
                  timeEst=timeEst,timeP=timeP,
                  timePolicyEst=timePolicyEst,timePolicyP=timePolicyP,N=N):
        
        policyEst.append(     regRes.params["After[T.True]"] )
        timeEst.append(       regRes.params["RecvMonth"] )
        timePolicyEst.append( regRes.params["After[T.True]:RecvMonth"] )
        
        policyP.append(     regRes.pvalues.loc['After[T.True]'])
        timeP.append(       regRes.pvalues.loc['RecvMonth'])
        timePolicyP.append( regRes.pvalues.loc['After[T.True]:RecvMonth'])
        
        N.append(regRes.nobs)
        
    def printRegRes(policyEst=policyEst,policyP=policyP,
                  timeEst=timeEst,timeP=timeP,
                  timePolicyEst=timePolicyEst,timePolicyP=timePolicyP,N=N):
        
        print(' '.join(["{:.2f}\%".format(i*100) for i in policyEst]))
        print(' '.join(["({:.3f})".format(i) for i in policyP]))
        
        print(' '.join(["{:.2f}\%".format(i*100) for i in timePolicyEst]))
        print(' '.join(["({:.3f})".format(i) for i in timePolicyP]))
        
        print(' '.join([str(i) for i in N]))
        
    ### values to be printed ###
    
    
    ### first row ###
    _, regRes = analysis(
        COIoverTime, x='RecvMonth', y='Less24', cutoff=139,
        RANGE=RANGE, ax=axes[0]
    ) # main
    if mainTable: addRegRes(regRes)
        
    _, regRes = analysis(
        COIoverTime, x='RecvMonth', y='Less48More24', cutoff=139,
        RANGE=RANGE, ax=axes[1], markersize=2
    ) # control
    if not mainTable: addRegRes(regRes)
        
    for ax in axes[:2]:
        xlabels = ['2007/7', '2009/7', '2011/7', '2013/7', '2015/7']
        ax.set_xticks([converMonth(datetime.strptime(x, '%Y/%m')) for x in xlabels])
        ax.set_xticklabels(xlabels)
    ### first row ###
        
    ### second row ###
    _, regRes = analysis(
        COIoverTime, x='RecvMonth', y='Less48More24', cutoff=169,
        RANGE=RANGE, ax=axes[2]
    ) # main one
    if mainTable: addRegRes(regRes)
        
    _, regRes = analysis(
        COIoverTime, x='RecvMonth', y='Less24', cutoff=169,
        RANGE=RANGE, ax=axes[3], markersize=2
    ) # less24 is control
    if not mainTable: addRegRes(regRes)

    for ax in axes[2:4]:
        xlabels = ['2010/1', '2012/1', '2014/1', '2016/1', '2018/1']
        ax.set_xticks([converMonth(datetime.strptime(x, '%Y/%m')) for x in xlabels])
        ax.set_xticklabels(xlabels)
    ### second row ###
    
    
    ### third row ###
    _, regRes = analysis(
        COIoverTimePLOS, x='RecvMonth', y='Less60', cutoff=185,
        RANGE=RANGE, ax=axes[4],
    )
    if mainTable: addRegRes(regRes)
        
    _, regRes = analysis(
        COIoverTimePLOS, x='RecvMonth', y='More60', cutoff=185, 
        RANGE=RANGE, ax=axes[5], markersize=2
    )
    if not mainTable: addRegRes(regRes)
        
    for ax in axes[4:]:
        xlabels = ['2011/5', '2013/5', '2015/5', '2017/5', '2019/5']
        ax.set_xticks([converMonth(datetime.strptime(x, '%Y/%m')) for x in xlabels])
        ax.set_xticklabels(xlabels)
    ### third row ###
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    axes[0].set_title('Most recent collaboration\n< 24 months ago', fontsize=7)
    axes[1].set_title('24 ~ 48 months ago (control group)', fontsize=7)
    
    axes[2].set_title('24 ~ 48 months ago', fontsize=7)
    axes[3].set_title('< 24 months ago (control group)', fontsize=7)
    
    axes[4].set_title('< 60 months ago', fontsize=7)
    axes[5].set_title('> 60 months ago (control group)', fontsize=7)
    
    ax11.set_ylim(0, 0.15)
    ax11.set_yticks([x/100 for x in range(0, 16, 5)])
    ax11.set_yticklabels([f'{x}%' for x in range(0, 16, 5)])
    
    ax12.set_ylim(0, 0.12)
    ax12.set_yticks([x/100 for x in range(0, 13, 4)])
    ax12.set_yticklabels([f'{x}%' for x in range(0, 13, 4)])
    
    ax21.set_ylim(0, 0.09)
    ax21.set_yticks([x/100 for x in range(0, 10, 3)])
    ax21.set_yticklabels([f'{x}%' for x in range(0, 10, 3)])
    
    ax22.set_ylim(0, 0.15)
    ax22.set_yticks([x/100 for x in range(0, 16, 5)])
    ax22.set_yticklabels([f'{x}%' for x in range(0, 16, 5)])
    
    ax31.set_ylim(0, 0.09)
    ax31.set_yticks([x/100 for x in range(0, 10, 3)])
    ax31.set_yticklabels([f'{x}%' for x in range(0, 10, 3)])
    
    ax32.set_ylim(0, 0.06)
    ax32.set_yticks([x/100 for x in range(0, 7, 2)])
    ax32.set_yticklabels([f'{x}%' for x in range(0, 7, 2)])
    
    
    printRegRes()