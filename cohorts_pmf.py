#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: myrthings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

from copy import deepcopy

from aux import *

def nums(data):
    data['period_num']=np.arange(len(data))
    #data['months']=data['months'].apply(lambda x: '+{:02}'.format(x))
    return data

def perc(data,name,column,column2=None):
    if not column2:
        column2=column
    data[name]=data[column]/data[column2].iloc[0]
    return data

def churn(data,name,column):
    data[name]=data[column].iloc[0]-data[column]
    return data

def accum(data,name,column):
    data[name]=data[column].cumsum()
    return data

class Cohorts(object):
    """
    Cohorts is an object implementation of the Cohorts Framework. The framework
    was developed by Tribe Capital on a Quantitative Approach towards Product Market Fit.
    
    This implementation uses PMF units data to fit the model and returns the Cohorts esquema
    previous to exploration. It also plots the data two different types of plots with 3 different
    approaches.
    
    We call PMF unit the unit you are using to measure Product Market Fit. Most common PMF unit
    are user interactions (simple) or revenue (not simple). Read 'Further thoughts and future directions' 
    on references article for more information.
    
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
    
    column_id : the string input in fit (see fit function)
    
    column_date : the string input in fit (see fit function)
    
    arguments : list of strings
        All the ways the model has been fitted and we can have a cohort representation on them.
        It include both standard and personalized ways, but it doesn't include the percentage
        because all arguments in this list have them.
        
    df_period_cohort : Pandas DataFrame
        It's the input dataframe with the 'period' and 'cohort' columns aded with the fit.
        It can be used to extract the 'period' and 'cohort' columns fitted by the model
        and calculate our own parameters to apply a personalized view of the data.
        
    df_cohorts : Pandas DataFrame with 3 + len(arguments)*2 columns
        It's the fitted dataframe grouped by cohorts and periods. It includes the column num_period
        and the arguments with its percentage.
    
    period_list: nparray of strings objects
        Gives a string tag to each period.
        
    cohort_list: nparray of strings objects
        Gives a string tag to each cohort.
    
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
            raise ValueError("Period should be one of those: m, q, d, 28d, 7d.")
        else:
            self.period=_period
            
        self.simple=simple
        self.arguments=[]
        
    def apply_unique_users(self,column_id):
        """
        Calculate the number un unique users on 'column_id' by cohort and period. Then, how the
        percentage varies compared to the first period of each cohort. It can only be executed
        after fit, because it needs both 'df_period_cohort' and 'df_cohorts' attributes.
        
        Parameters
        ----------
        column_id : string
            Name of the column on 'df_period_cohorts' with the unique users ids.
        
        Returns
        -------
        self
            Adds a columns 'unique_users' and 'perc_unique_users' to 'df_cohorts' attribute.

        
        if not self.df_cohorts:
            raise ValueError('This function can only be executed after fit.')
        """
        
        df=self.df_period_cohort
        df_cohorts=self.df_cohorts.set_index(['cohort','period'])
                
        df_cohorts['unique_users']=df.groupby(['cohort','period'])[column_id].nunique()
        df_cohorts=df_cohorts.reset_index().groupby('cohort').apply(perc,name='perc_{}'.format('unique_users'),column='unique_users')
                
        self.df_cohorts=df_cohorts
        
        
    def apply_total(self,column_input):
        """
        Calculate the number of total PMF units on 'column_input' by cohort and period. Then, how the
        percentage varies compared to the first period of each cohort. It can only be executed
        after fit, because it needs both 'df_period_cohort' and 'df_cohorts' attributes.
        
        Parameters
        ----------
        column_input : string
            Name of the column on 'df_period_cohorts' with the values to add.
        
        Returns
        -------
        self
            Adds a columns 'total' and 'perc_total' to 'df_cohorts' attribute.

        
        if not self.df_cohorts:
            raise ValueError('This function can only be executed after fit.')
        """
        
        df=self.df_period_cohort
        df_cohorts=self.df_cohorts.set_index(['cohort','period'])
                
        df_cohorts['total']=df.groupby(['cohort','period'])[column_input].sum()
        df_cohorts=df_cohorts.reset_index().groupby('cohort').apply(perc,name='perc_{}'.format('total'),column='total')
                
        self.df_cohorts=df_cohorts
        
    
    def apply_churn_total(self,column_input):
        """
        Calculate the churn of PMF units on 'column_input' by cohort and period. Then, how the
        percentage varies compared to the total on first period of each cohort. It can only be executed
        after fit, because it needs both 'df_period_cohort' and 'df_cohorts' attributes.
        
        If 'total' is not on 'df_cohorts', it computes it.
        
        Parameters
        ----------
        column_input : string
            Name of the column on 'df_period_cohorts' with the values to add.
        
        Returns
        -------
        self
            Adds a columns 'churn_total' and 'perc_churn_total' to 'df_cohorts' attribute.

        
        if not self.df_cohorts:
            raise ValueError('This function can only be executed after fit.')
        """
        
        if 'total' not in self.df_cohorts.columns:
            self.apply_total(column_input)
        
        df_cohorts=self.df_cohorts
                
        df_cohorts=df_cohorts.groupby('cohort').apply(churn,name='churn_total',column='total')
        df_cohorts=df_cohorts.groupby('cohort').apply(perc,name='perc_{}_total'.format('churn'),column='churn_total',column2='total')
        
        self.df_cohorts=df_cohorts
        
    def apply_churn_unique_users(self,column_id):
        """
        Calculate the churn of unique users on 'column_ids' by cohort and period. Then, how the
        percentage varies compared to the unique_users on first period of each cohort. It can only be executed
        after fit, because it needs both 'df_period_cohort' and 'df_cohorts' attributes.
        
        If 'unique_users' is not 'df_cohorts', it computes it.
        
        Parameters
        ----------
        column_id : string
            Name of the column on 'df_period_cohorts' with the user ids.
        
        Returns
        -------
        self
            Adds a columns 'churn_unique' and 'perc_churn_unique' to 'df_cohorts' attribute.

        
        if not self.df_cohorts:
            raise ValueError('This function can only be executed after fit.')
        """
        
        
        if 'unique_users' not in self.df_cohorts.columns:
            self.apply_unique(column_id)
        
        df_cohorts=self.df_cohorts
                
        df_cohorts=df_cohorts.groupby('cohort').apply(churn,name='churn_unique',column='unique_users')
        df_cohorts=df_cohorts.groupby('cohort').apply(perc,name='perc_{}_unique'.format('churn'),column='churn_unique',column2='unique_users')
        
        self.df_cohorts=df_cohorts
        
    def apply_accum(self,column_input):
        """
        Calculate the cumulative value of PMF units on 'column_input' by cohort and period. Then, how the
        percentage varies compared to the total on first period of each cohort. It can only be executed
        after fit, because it needs both 'df_period_cohort' and 'df_cohorts' attributes.
        
        If 'total' is not in 'df_cohorts', it computes it.
        
        Parameters
        ----------
        column_input : string
            Name of the column on 'df_period_cohorts' with the values to add.
        
        Returns
        -------
        self
            Adds a columns 'accum' and 'perc_accum' to 'df_cohorts' attribute.
            
        Note
        ----
        It can be used to calculate LTV. The name is more generic for generalization purposes.

        
        if not self.df_cohorts:
            raise ValueError('This function can only be executed after fit.')
        """
        
        if 'total' not in self.df_cohorts.columns:
            self.apply_total(column_input)
        
        df_cohorts=self.df_cohorts
                
        df_cohorts=df_cohorts.groupby('cohort').apply(accum,name='accum',column='total')
        df_cohorts=df_cohorts.groupby('cohort').apply(perc,name='perc_{}'.format('accum'),column='accum',column2='total')
        
        self.df_cohorts=df_cohorts
        
        
    def apply_per_user(self,column_input,column_id):
        """
        Calculate the value of PMF units on 'column_input' per 'column_id' by cohort and period. Then, how the
        percentage varies compared to the one on first period of each cohort. It can only be executed
        after fit, because it needs both 'df_period_cohort' and 'df_cohorts' attributes.
        
        If 'total' and 'unique_users' are not in 'df_cohorts', it computes them.
        
        Parameters
        ----------
        column_input : string
            Name of the column on 'df_period_cohorts' with the values to add.
            
        column_id : string
            Name of the column on 'df_period_cohorts' with the user ids.
        
        Returns
        -------
        self
            Adds a columns 'per_user' and 'perc_per_user' to 'df_cohorts' attribute.
            
        Note
        ----
        It can be used to calculate use frequencies or revenue per user. The name is more generic for
        generalization purposes.

        
        if not self.df_cohorts:
            raise ValueError('This function can only be executed after fit.')
        """
        
        if 'total' not in self.df_cohorts.columns:
            self.apply_total(column_input)
            
        if 'unique_users' not in self.df_cohorts.columns:
            self.apply_unique_users(column_id)
            
        df_cohorts=self.df_cohorts
        df_cohorts['per_user']=df_cohorts['total']/df_cohorts['unique_users']
        df_cohorts=df_cohorts.groupby('cohort').apply(perc,name='perc_{}'.format('per_user'),column='per_user')
        
        self.df_cohorts=df_cohorts
        
    def apply_personalized(self,df_cohorts,column_label):
        """
        Add a personalized label by cohort and period. Then, how it calculates how the
        percentage varies compared to the one on first period of each cohort. It can only be executed
        after fit, because it needs both 'df_period_cohort' and 'df_cohorts' attributes.
                
        Parameters
        ----------
        df_cohort : Pandas DataFrame as the same kind of 'df_cohorts' attribute.
            It should include 'period', 'cohort' and a column labeled with the argument to include in the model.
            
        column_label : string
            Name of the column on 'df_cohorts' parameter with the new argument to add.
        
        Returns
        -------
        self
            Adds a columns column_label and perc_{column_label} to 'df_cohorts' attribute.
            Adds the new parameter to 'arguments' attribute.
            
        Note
        ----
        It can be used to calculate for example revenue per order or variety, or any kind of data
        wrangling not included as an apply function.

        
        if not self.df_cohorts:
            raise ValueError('This function can only be executed after fit.')
        """
        
        df_cohorts=pd.merge(self.df_cohorts,df_cohorts[['cohort','period',column_label]],on=['cohort','period'],how='outer')
        df_cohorts=df_cohorts.groupby('cohort').apply(perc,name='perc_{}'.format(column_label),column=column_label)
        
        self.df_cohorts=df_cohorts
        
        _args=self.arguments
        _args.append(column_label)
        _args=set(_args)
        self.arguments=list(_args)
        
        
    def fit(self,data,column_date,column_id,column_input=None,how=[]):
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
            
        how: list of strings on {'total','churn_total','accum','unique_users',
            'churn_unique_users','accum'} or empty, default=[]
            Methods to calculate the cohorts we want.
        
        Returns
        -------
        self
            Fitted object.

        """
        if not self.simple and not column_input:
            raise ValueError("simple = False requires argument 'column_input'")
        else:
            df=deepcopy(data)
            df['unique_id']=np.arange(len(df))
            
            if self.simple:
                column_input='column_input'
                df[column_input]=1
                
            df['period']=df[column_date].apply(lambda x: custom_representative(self.period,x))

            df.set_index(column_id,inplace=True)
            df['cohort']=df.groupby(level=0)['period'].min()
            df=df.reset_index()
            
            
            self.df_period_cohort=df
            
            cohorts=df.groupby(['cohort','period'])[[column_input]].count().rename(columns={column_input:'centinela'})
            cohorts=cohorts.unstack().fillna(0).stack().reset_index()
            cohorts=cohorts[cohorts['cohort']<=cohorts['period']]
            cohorts=cohorts.groupby('cohort').apply(nums)
            
            self.period_list=list(cohorts['period'].apply(lambda x: custom_period(self.period,x)).unique())
            self.cohort_list=list(cohorts['cohort'].apply(lambda x: custom_period(self.period,x)).unique())
            
            self.df_cohorts=cohorts[['cohort','period','period_num']]
                        
            dic={'total': self.apply_total,
                 'churn_total': self.apply_churn_total,
                 'accum': self.apply_accum,
                 'unique_users': self.apply_unique_users,
                 'churn_unique_users':self.apply_churn_unique_users,
                 'per_user':self.apply_per_user}
            
            for item in how:
                if item in ['total','churn_total','accum']:
                    dic[item](column_input)
                elif item in ['unique_users','churn_unique_users']:
                    dic[item](column_id)
                elif item=='per_user':
                    dic[item](column_input,column_id)
                else:
                    raise ValueError("How values should be inside some of these: 'total','churn_total','accum','unique_users','churn_unique_users','per_user'")
            
            self.arguments=list(set(how+self.arguments))
            self.column_id=column_id
            self.column_date=column_date
            
    def transform_pd(self,label,way='period'):
        """
        Returns the cohort matrix of the label givne by the way given in a Pandas DataFrame.
                
        Parameters
        ----------
        label : string in 'arguments' attribute or perc of those
            Used to calculate the cohorts matrix.
        
        way : string in {'period','period_num'}, default='period'
            Used to calculate the cohorts matrix with this parameter

        Returns
        -------
        pandas DataFrame with the cohorts as index and 'way' as columns containing the label values

        """
        df_cohorts=self.df_cohorts.set_index(['cohort',way])[[label]].unstack()
        return df_cohorts
    
    def transform_np(self,label,way='period'):
        """
        Returns the cohort matrix of the label given by the way given as a Numpy Array.
                
        Parameters
        ----------
        label : string in 'arguments' attribute or perc of those
            Used to calculate the cohorts matrix.
        
        way : string in {'period','period_num'}, default='period'
            Used to calculate the cohorts matrix with this parameter

        Returns
        -------
        list_periods : a nparray with strings of the periods/period_nums used as columns labels
        
        list_cohorts : a nparray with strings of the cohorts used as row labels
        
        numpy_cohorts : numpy arry of shape (len(way),len(periods))
        """
        numpy_cohorts=np.array(self.df_cohorts.set_index(['cohort',way])[[label]].unstack())
        list_periods=np.array(self.df_cohorts[way].unique())
        list_cohorts=np.array(self.df_cohorts['cohort'].unique())
        return list_periods,list_cohorts,numpy_cohorts
    
    
    def plot_heatmap(self,label,title,way='period'):
        """
        Plot the Cohorts given the label and the way. It includes a barchart with the cohort size
        and the heatmap with the values. For percentage labels the colors are fixed between 0% and 200%
        for visualization sake.
        
        Parameters
        ----------
        label : string in 'arguments' attribute or perc of those
            Used to calculate the cohorts matrix.
            
        title : string
            Used to give a title to the matplotlib figure
            
        way : string in {'period','period_num'}, default='period'
            Used to calculate the cohorts matrix with this parameter
        
        Returns
        -------
        figure
            It directly plots the values on a matplolib figure

        """
        ## prepare the data
        if 'unique_users' not in self.df_cohorts.columns:
            self.apply_unique_users(self.column_id)
            
        cohort_size=self.df_cohorts.groupby('cohort')[['unique_users']].apply(lambda x: x.iloc[0]).rename(columns={'unique_users':'cohort_size'})
        
        df_coh=self.df_cohorts.set_index(['cohort',way])[label].unstack()
        
        ## prepare the plot
        max_coh=round(len(self.df_cohorts['cohort'].unique())*0.75)
        max_per=round(len(self.df_cohorts['period'].unique())*1.5)
        
        fig,(ax1,ax2)=plt.subplots(1,2,gridspec_kw={'width_ratios': [1, round(max_per/3) if max_per>2 else 3]},figsize=(max_per,max_coh))
        
        ## plot
        cohort_size['cohort_size'].sort_index(ascending=False).plot(kind='barh',width=0.9,color='grey',alpha=0.5,ax=ax1)
        if label[:4]=='perc':
            ax=sns.heatmap(df_coh,
                       cmap='coolwarm_r',center=1,vmin=0, vmax=2,
                       annot=True,fmt='1.0%',
                       ax=ax2)
        elif 'unique' in label:
            ax=sns.heatmap(df_coh,
                       cmap='coolwarm_r',
                       annot=True,fmt='.0f',
                       ax=ax2)
        elif self.df_cohorts[label].mean()<5:
            ax=sns.heatmap(df_coh,
                       cmap='coolwarm_r',
                       annot=True,fmt='.3f',
                       ax=ax2)
        else:
            ax=sns.heatmap(df_coh,
                       cmap='coolwarm_r',
                       annot=True,fmt='.0f',
                       ax=ax2)
        
        #better axes
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.75, top - 0.75)
        labels=ax2.get_yticklabels()
        #ax2.set_yticklabels(labels,fontsize=14,rotation=0)
        ax2.set_yticklabels('')
        ax2.set_ylabel('')
        
        if way=='period_num':
            ax2.set_xticklabels([0,'+1 period']+['+{}'.format(x) for x in range(2,self.df_cohorts[way].max()+1)],fontsize=12)
        else:
            ax2.set_xticklabels(self.period_list)
        ax2.set_xlabel(way)

        ax1.set_xlim(xmin=0)
        ax1.set_yticklabels(pd.Series(self.cohort_list).sort_values(ascending=False))
        ax1.set_ylabel('')
        
        ax1.set_title('Cohort size',fontsize=16)
        ax2.set_title(title,fontsize=16)

        fig.tight_layout()
        
        plt.show()
        
    def plot_trends(self,label,title,way='period'):
        """
        Plot the Trends given the label and the way.
        
        Parameters
        ----------
        label : string in 'arguments' attribute or perc of those
            Used to calculate the cohorts matrix.
            
        title : string
            Used to give a title to the matplotlib figure
            
        way : string in {'period','period_num','age'}, default='period'
            Used to calculate the cohorts matrix with this parameter. The parameter 'age'
            means every line ploted is a 'period_num' and the x-axis are the cohorts.
            Parameters 'period' and 'period_num' means each line is a cohort and the x-axis
            is both one or the other.
        
        Returns
        -------
        figure
            It directly plots the values on a matplolib figure

        """
        df_cohorts=self.df_cohorts
        # 'age' means every line is a period num and cohorts are on the x-axis
        if way=='age':
            for num in df_cohorts['period_num'].unique():
                dat=df_cohorts.loc[df_cohorts['period_num']==num,['cohort',label]]
                plt.plot(dat['cohort'],dat[label],label=num)
                
        #age means every line is a period num and cohorts are on the x-axis
        else:
        
            for i,cohort in enumerate(df_cohorts['cohort'].unique()):
                dat=df_cohorts.loc[df_cohorts['cohort']==cohort,[way,label]]
                plt.plot(dat[way],dat[label],label=self.cohort_list[i])

            plt.plot(df_cohorts[way].unique(),df_cohorts.set_index(['cohort',way])[[label]].unstack().mean().values,'k--',label='mean')
            
        if label[:4]=='perc':
            ticks,labels=plt.yticks()
            plt.yticks(ticks,['{:.0%}'.format(t) for t in ticks])
            
        plt.title(title)
        plt.legend()
        plt.show()
        
        
        
        
