#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: myrthings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from copy import deepcopy

from aux import *


class GrowthAccounting(object):
    """
    GrowthAccounting is an object implementation of the Growth Accounting Framework. The framework
    was developed by Tribe Capital on a Quantitative Approach towards Product Market Fit.
    
    This implementation uses PMF units data to fit the model and returns the Growth Accounting
    attributes. It also plots the data in two different ways to help quick lookups to the data.
    
    We call PMF unit the unit you are using to measure Product Market Fit. Most common PMF unit
    are user interactions or revenue. Read 'Further thoughts and future directions' on references 
    article for more information.
    
    Parameters
    ----------
    period : {'Q','q','M','m','28D','28d','7D','7d','D','d'}, default='M'
        Used to specify the period in which to divde the data. It supports both cased or uncased
        parameters.
         - 'Q' means quarter of a year /3 months.
         - 'M' means monthly.
         - '28D' means 4 weeks starting on Monday and it's used as an alterantive to months to fix 
            the number of days for the period.
         - '7D' means weekly starting on Monday.
         - 'D' refers to daily.
    
    simple : bool, default=True
        Used to specify the kind of data that will fit the object. True means each row will be
        a PMF unit (ex. user interactions) and we only need the 'user_id' and the 'date' of the 
        interaction to build the framework. False means each row represent a quantity of the PMF 
        unit (ex. revenue) and we need the columns on simple but also a 'quantity' column to 
        build the framework.
        
    
    Attributes
    ----------
    period : the period input (see on Parameters section)
    
    simple : the simple input (see on Parameters section)
    
    period_rep_list : nparray of datetime objects
        Each period ocurrs between the date in this list and the date in this list + period (attribute).
        It uses the first day as the representative of the whole period.
    
    period_list: nparray of strings objects
        Gives a string tag to each period. It's more knowledgable than period_rep_list but
        worse to use directly on charts because of its string nature.
        
    total : nparray of floats of length len(period_list)
        Represents total PMF unit per period.
        
    new : nparray of floats of length len(period_list)
        Represents new PMF unit per period.
        
    resurrected : nparray of floats of length len(period_list)
        Represents resurrected PMF unit per period.
        
    expansion : nparray of floats of length len(period_list)
        Represents expansion PMF unit per period.
        
    contraction : nparray of floats of length len(period_list)
        Represents contraction PMF unit per period.
        
    retained : nparray of floats of length len(period_list)
        Represents retained PMF unit per period.
        
    churned : nparray of floats of length len(period_list)
        Represents churned PMF unit per period.
        
    new_rate : nparray of floats of length len(period_list)
        Represents the change in new PMF unit per period.
        
    resurrected_rate : nparray of floats of length len(period_list)
        Represents the change in resurrected PMF unit per period.
        
    expansion_rate : nparray of floats of length len(period_list)
        Represents the change in expansion PMF unit per period.
        
    contraction_rate : nparray of floats of length len(period_list)
        Represents the change in contraction PMF unit per period.
        
    retained_rate : nparray of floats of length len(period_list)
        Represents the change in retained PMF unit per period.
        
    churned_rate : nparray of floats of length len(period_list)
        Represents the change in churned PMF unit per period.
        
    growth_rate : nparray of floats of length len(period_list)
        Represents Growth Rate of PMF unit per period.
        
    gross_retention : nparray of floats of length len(period_list)
        Represents Gross Retention of PMF unit per period.
    
    quick_ratio : nparray of floats of length len(period_list)
        Represents Quick Ratio of PMF unit per period.
    
    net_churn : nparray of floats of length len(period_list)
        Represents Net Churn of PMF unit per period.
        
    df : Pandas DataFrame with len(period_list) rows and 16+ columns.
        Is a dataframe containing at least all parameters above from 'total' to 'net_churn'
        that describes the PMF unit behavior in this framework. It's said at least because if
        'compound_growth' is executed it also add that to the dataframe.
    
    Detailed Use Example
    --------------------
    (future link to a jupyter notebook implementation)
    
    References
    ----------
    Article with the framework:
        https://tribecap.co/a-quantitative-approach-to-product-market-fit/
    
    """
    
    def __init__(self,period='M',simple=True):
        _period=period.lower()
        
        if _period not in ['m','q','d','28d','7d']:
            raise ValueError("Period paremeter should be in: {'m', 'q', 'd', '28d', '7d'}")
        else:
            self.period=_period
            
        self.simple=simple
        
    
    def fit(self,data,column_date,column_id,column_input=None):
        """
        Fit the model with the given data.
        
        Parameters
        ----------
        data : Pandas DataFrame with at least 2 columns (or 3 in the case of not simple)
            Each row in this DataFrame should represent and action of PMF unit. We can use both
            actions as they are (simple) or with a quantity column (not simple)
            
        column_date : string
            Name of the column on 'data' with inputs in datetime format.
            
        column_id : string
            Name of the column on 'data' containing the ids of the consumers / users.
            
        column_input : string, default=None
            Name of the column on 'data' containing the quantinty of the PMF unit.
            Default is None because 'simple' parameter is True by default.
        
        Returns
        -------
        self
            Fitted object.

        """
        
        if not self.simple and not column_input:
            raise ValueError("simple = False requires argument 'column_input'")
        
        else:
            print('Preparing the data...')
            df=deepcopy(data)
            df['unique_id']=np.arange(len(df))
            
            if self.simple:
                column_input='column_input'
                df[column_input]=1
                
            df['period_rep']=df[column_date].apply(lambda x: custom_representative(self.period,x))

            df.set_index(column_id,inplace=True)
            df['cohort_rep']=df.groupby(level=0)['period_rep'].min()
            df=df.reset_index()
            
            df_users=df.groupby([column_id,'cohort_rep','period_rep'])[['unique_id']].count().rename(columns={'unique_id':'freq'})
            df_users=df_users.unstack().fillna(0).stack().reset_index()
            df_users=df_users[df_users['period_rep']>=df_users['cohort_rep']]
            df_users['freq']=df_users['freq'].astype(int)
            df_users['uniques']=df_users['freq'].apply(lambda x: np.where(x>0,1,0))

            def user_change(df):
                df['change']=df['uniques'].diff()
                return df

            df_users=df_users.groupby([column_id,'cohort_rep']).apply(user_change)
            df_users['change']=df_users['change'].fillna(2)
            df_users['supercolumn']=df_users['change']+df_users['uniques']
            
            df_revenue=df_users[[column_id,'cohort_rep','period_rep','supercolumn']].set_index([column_id,'period_rep'])
            df_revenue['revenue']=df.groupby([column_id,'period_rep'])[column_input].sum()
            df_revenue['revenue']=df_revenue['revenue'].fillna(0)
            df_revenue=df_revenue.reset_index()

            def revenue_change(df):
                df['revchange']=df['revenue'].diff()
                return df

            df_revenue=df_revenue.groupby([column_id,'cohort_rep']).apply(revenue_change)
            df_revenue['revstatus']=df_revenue['revchange'].apply(lambda x: np.where(x>0,1,np.where(x<0,-1,0)))
            df_revenue['retained']=df_revenue['revenue']-df_revenue['revchange'].apply(lambda x: np.where(x<0,0,x))
            
            print('Computing Growth Accounting...')
            dfgrowth=df.groupby(['cohort_rep'])[[column_id]].nunique().rename(columns={column_id:'new_ids'})
            dfgrowth['total_ids']=df.groupby(['period_rep'])[[column_id]].nunique()
            dfgrowth['total_orders']=df.groupby(['period_rep'])[[column_id]].count()
            
            dfgrowth['new']=df_revenue[df_revenue['supercolumn']==3].groupby('period_rep')['revenue'].sum()
            dfgrowth['resurrected']=df_revenue[df_revenue['supercolumn']==2].groupby('period_rep')['revenue'].sum()
            dfgrowth['expansion']=df_revenue[(df_revenue['supercolumn']==1)&(df_revenue['revstatus']==1)].groupby('period_rep')['revchange'].sum()
            dfgrowth['contraction']=df_revenue.loc[(df_revenue['supercolumn']==1)&(df_revenue['revstatus']==-1)].groupby('period_rep')['revchange'].sum()
            dfgrowth['retained']=df_revenue.loc[(df_revenue['supercolumn']==1)].groupby('period_rep')['retained'].sum()
            dfgrowth['churned']=df_revenue.loc[(df_revenue['supercolumn']==-1)].groupby('period_rep')['revchange'].sum()

            dfgrowth=dfgrowth.fillna(0)
            
            dfgrowth['total']=df_revenue.groupby('period_rep')['revenue'].sum()
            dfgrowth['total_rev2']=dfgrowth['new']+dfgrowth['retained']+dfgrowth['expansion']+dfgrowth['resurrected']
            dfgrowth['total_rev-1']=dfgrowth['retained']-dfgrowth['churned']-dfgrowth['contraction']
            
            
            dfgrowth['new_rate']=dfgrowth['new']/dfgrowth['new'].shift(1)
            dfgrowth['resurrected_rate']=dfgrowth['resurrected']/dfgrowth['resurrected'].shift(1)
            dfgrowth['expansion_rate']=dfgrowth['expansion']/dfgrowth['expansion'].shift(1)
            dfgrowth['contraction_rate']=dfgrowth['contraction']/dfgrowth['contraction'].shift(1)
            dfgrowth['retained_rate']=dfgrowth['retained']/dfgrowth['retained'].shift(1)
            dfgrowth['churned_rate']=dfgrowth['churned']/dfgrowth['churned'].shift(1)
            
            dfgrowth['growth_rate']=dfgrowth['new_rate']+dfgrowth['resurrected_rate']+dfgrowth['expansion_rate']-dfgrowth['contraction_rate']-dfgrowth['churned_rate']
            dfgrowth['gross_retention']=dfgrowth['retained']/dfgrowth['total'].shift(1)
            dfgrowth['quick_ratio']=(dfgrowth['new']+dfgrowth['resurrected']+dfgrowth['expansion'])/(-dfgrowth['churned']-dfgrowth['contraction'])
            dfgrowth['net_churn']=(-dfgrowth['churned']-dfgrowth['contraction']-dfgrowth['resurrected']-dfgrowth['expansion'])/(dfgrowth['total'].shift(1))
            
            dfgrowth=dfgrowth.replace(np.inf, np.nan).replace(-np.inf, np.nan)
            # dfgrowth=dfgrowth.fillna(0) #this falsifies the data
            
            self.period_rep_list=np.array(dfgrowth.index)
            self.period_list=np.array(pd.Series(dfgrowth.index).apply(lambda x: custom_period(self.period,x)))
            
            self.total=np.array(dfgrowth['total'])
            self.new=np.array(dfgrowth['new'])
            self.resurrected=np.array(dfgrowth['resurrected'])
            self.expansion=np.array(dfgrowth['expansion'])
            self.contraction=np.array(dfgrowth['contraction'])
            self.retained=np.array(dfgrowth['retained'])
            self.churned=np.array(dfgrowth['churned'])
            
            self.new_rate=np.array(dfgrowth['new_rate'])
            self.resurrected_rate=np.array(dfgrowth['resurrected_rate'])
            self.expansion_rate=np.array(dfgrowth['expansion_rate'])
            self.contraction_rate=np.array(dfgrowth['contraction_rate'])
            self.retained_rate=np.array(dfgrowth['retained_rate'])
            self.churned_rate=np.array(dfgrowth['churned_rate'])
            
            self.growth_rate=np.array(dfgrowth['growth_rate'])
            self.gross_retention=np.array(dfgrowth['gross_retention'])
            self.quick_ratio=np.array(dfgrowth['quick_ratio'])
            self.net_churn=np.array(dfgrowth['net_churn'])
            
            self.df=dfgrowth[['total','new','resurrected','expansion','contraction','retained','churned',
                             'new_rate','resurrected_rate','expansion_rate','retained_rate','churned_rate',
                             'growth_rate','gross_retention','quick_ratio','net_churn']]
            
            print('Done!')
            
    def compound_growth(self,num_periods):
        """
        Calculate the Compound Growth on the period size, given a number of periods.
        
        Parameters
        ----------
        num_periods : int
            Number of periods to calculate the compound growth.
        
        Returns
        -------
        self
            Adds a column with the value to the 'df' attribute.
            
        np array of floats with the values

        """
        label='C{}GR{}'.format(self.period,num_periods).upper()
        self.df[label]=(self.df['total']/self.df['total'].shift(num_periods)).apply(lambda x: x**(1/num_periods)-1)
        return np.array(self.df[label])
    
    def plot_compound_growth(self,num_periods):
        """
        Calculate and plots the Compound Growth on the period size, given a number of periods.
        
        Parameters
        ----------
        num_periods : int
            Number of periods to calculate the compound growth.
        
        Returns
        -------
        self
            Adds a column with the value to the 'df' attribute.
            
        figure
            It directly plots the values on a matplolib figure

        """
        label='C{}GR{}'.format(self.period,num_periods).upper()
        self.df[label]=(self.df['total']/self.df['total'].shift(num_periods)).apply(lambda x: x**(1/num_periods)-1)
        comp = np.array(self.df[label])    
    
        fig,ax1=plt.subplots(1,1,figsize=(15,5))
        ax2=ax1.twinx() #sacar eje secundario
        
        ax1.bar(self.period_list,self.total,color='blue',width=1,alpha=0.5,label='Revenue')
        ax2.plot(self.period_list,comp,color='black',label=label)
        
        fig.legend()
        plt.show()
    
    def create_plot(self,ax1):
        """
        Creates the plot for the Growth Accounting Framework. It can be used to personalize
        later the plot with custom title, legend, axlabels or whatever.
        
        Parameters
        ----------
        ax1 : matplotlib figure ax
            Used to plot the charts and to get a duplicated axis as ax2
        
        Returns
        -------
        ax1 : matplotlib figure ax
            With the elements ploted
            
        ax2 : matplotlib twinx of ax1.
            With the elements ploted

        """
        ax1.bar(self.period_list,self.new,color='blue',width=1,alpha=0.2,label='New')
        ax1.bar(self.period_list,self.resurrected,bottom=self.new,color='blue',width=1,alpha=0.5,label='Resurrected')
        ax1.bar(self.period_list,self.expansion,bottom=self.new+self.resurrected,color='blue',width=1,alpha=0.7,label='Expansion')
        
        ax1.bar(self.period_list,self.churned,color='red',width=1,alpha=0.2,label='Churned')
        ax1.bar(self.period_list,self.contraction,bottom=self.churned,color='red',width=1,alpha=0.5,label='Contraction')
        
        ax2=ax1.twinx()
        ax2.plot(self.period_list,self.gross_retention,color='blue',label='Gross Retention')
        ax2.plot(self.period_list,self.net_churn,color='black',label='Net Churn')
        ax2.plot(self.period_list,self.quick_ratio,color='green',label='Quick Ratio')
        

        #arreglamos los ticks
        ticks1=ax1.get_yticks()
        ax1.set_ylim(ymin=ticks1.min(),ymax=ticks1.max())

        ticks1=ax1.get_yticks()
        ticks2=ax2.get_yticks()
        ax2.set_ylim(ymax=ticks2.max(),ymin=(ticks2.max()*ticks1.min())/ticks1.max())

        ticks=ax1.get_xticks()
        ax1.set_xlim(xmin=min(ticks)-0.5,xmax=max(ticks)+0.5)

        ax1.hlines(0,min(ticks)-1,max(ticks)+1,linewidth=0.5,alpha=0.5)
        
        return ax1,ax2
        
    
    def plot(self):
        """
        Plot the Growth Accounting Framework.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        figure
            It directly plots the values on a matplolib figure

        """
        fig,ax1=plt.subplots(1,1,figsize=(10,8))
        
        #creates the plot
        ax1,ax2=self.create_plot(ax1)
        fig.legend()
        
        plt.show()