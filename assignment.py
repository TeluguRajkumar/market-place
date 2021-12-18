#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 08:36:07 2021

@author: rajkumar
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('/Users/rajkumar/Desktop/df.xlsx')
df.head()
df.info()
df['HeroProduct1'].value_counts()
df['HeroProduct2'].value_counts()
df['HeroProduct1'].value_counts(normalize=True)
df['HeroProduct2'].value_counts(normalize=True)
# df.count() does not include NaN values
df2 = df[[column for column in df if df[column].count() / len(df) >= 0.3]]
del df2['DateAdded']
print("List of dropped columns:", end=" ")
for c in df.columns:
    if c not in df2.columns:
        print(c, end=", ")
print('\n')
df = df2
print(df['HeroProduct1'].describe())
print(df['HeroProduct2'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['HeroProduct1'], color='g', bins=100, hist_kws={'alpha': 0.4});
sns.distplot(df['HeroProduct2'], color='g', bins=100, hist_kws={'alpha': 0.4});

##Numerical data distribution
list(set(df.dtypes.tolist()))
df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num.head()
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); # ; avoid having the matplotlib verbose informations

##Correlation
df_num_corr = df_num.corr()['HeroProduct1'][:-1] # -4 because thar row is Heroproduct1
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with HeroProduct1:\n{}".format(len(golden_features_list), golden_features_list))
df_num_corr = df_num.corr()['HeroProduct2'][:-1] # -3 because that row is HeroProduct2
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with HeroProduct2:\n{}".format(len(golden_features_list), golden_features_list))
for i in range(0, len(df_num.columns), 5):
    sns.pairplot(data=df_num,
                x_vars=df_num.columns[i:i+5],
                y_vars=['HeroProduct1'])
for i in range(0, len(df_num.columns), 5):
    sns.pairplot(data=df_num,
                x_vars=df_num.columns[i:i+5],
                y_vars=['HeroProduct2'])

import operator

individual_features_df = []
for i in range(0, len(df_num.columns) - 4): # -4 because the  column is HerpProduct1
    tmpDf = df_num[[df_num.columns[i], 'HeroProduct1']]
    tmpDf = tmpDf[tmpDf[df_num.columns[i]] != 0]
    individual_features_df.append(tmpDf)

all_correlations = {feature.columns[0]: feature.corr()['HeroProduct1'][0] for feature in individual_features_df}
all_correlations = sorted(all_correlations.items(), key=operator.itemgetter(1))
for (key, value) in all_correlations:
    print("{:>15}: {:>15}".format(key, value))
    
individual_features_df = []
for i in range(0, len(df_num.columns) - 3): # -3 because the  column is HerpProduct2
    tmpDf = df_num[[df_num.columns[i], 'HeroProduct2']]
    tmpDf = tmpDf[tmpDf[df_num.columns[i]] != 0]
    individual_features_df.append(tmpDf)

all_correlations = {feature.columns[0]: feature.corr()['HeroProduct2'][0] for feature in individual_features_df}
all_correlations = sorted(all_correlations.items(), key=operator.itemgetter(1))
for (key, value) in all_correlations:
    print("{:>15}: {:>15}".format(key, value))
    
golden_features_list = [key for key, value in all_correlations if abs(value) >= 0.5]
print("There is {} strongly correlated values with HeroProduct1:\n{}".format(len(golden_features_list), golden_features_list))
golden_features_list = [key for key, value in all_correlations if abs(value) >= 0.5]
print("There is {} strongly correlated values with HeroProduct2:\n{}".format(len(golden_features_list), golden_features_list))
        
##Feature to feature relationship    
corr = df_num.drop('HeroProduct1', axis=1).corr() # We already examined SalePrice correlations
plt.figure(figsize=(12, 10))
sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);
corr = df_num.drop('HeroProduct2', axis=1).corr() # We already examined SalePrice correlations
plt.figure(figsize=(12, 10))
sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);

###Q -> Q (Quantitative to Quantitative relationship)¶
quantitative_features_list = [ 'category', 'sellerlink', 'sellerlinkurl', 'sellerstorefronturl', 'sellerproductcount', 'sellerratings', 'sellerdetails', 'sellerbusinessname', 'businessaddress', 'Countofsellerbrands', 'Countofsellerbrands', 'Max%ofnegativesellerratingslast90days', 'Max%ofnegativesellerratingslast12months', 'HeroProduct1', 'HeroProduct2', 'Samplebrandname', 'SampleBrandURL' ]
df_quantitative_values = df[quantitative_features_list]
df_quantitative_values.head()
features_to_analyse = [x for x in quantitative_features_list if x in golden_features_list]
features_to_analyse.append('HeroProduct1')
features_to_analyse
features_to_analyse = [x for x in quantitative_features_list if x in golden_features_list]
features_to_analyse.append('HeroProduct2')
features_to_analyse
fig, ax = plt.subplots(round(len(features_to_analyse) / 3), 3, figsize = (18, 12))

for i, ax in enumerate(fig.axes):
    if i < len(features_to_analyse) - 1:
        sns.regplot(x=features_to_analyse[i],y='HeroProduct1', data=df[features_to_analyse], ax=ax)
for i, ax in enumerate(fig.axes):
    if i < len(features_to_analyse) - 1:
        sns.regplot(x=features_to_analyse[i],y='HeroProduct2', data=df[features_to_analyse], ax=ax)

##C -> Q (Categorical to Quantitative relationship)
# quantitative_features_list[:-4] as that column is HeroProduct1 and we want to keep it
categorical_features = [a for a in quantitative_features_list[:-1] + df.columns.tolist() if (a not in quantitative_features_list[:-4]) or (a not in df.columns.tolist())]
df_categ = df[categorical_features]
df_categ.head()
# quantitative_features_list[:-3] as that column is HeroProduct2 and we want to keep it
categorical_features = [a for a in quantitative_features_list[:-3] + df.columns.tolist() if (a not in quantitative_features_list[:-4]) or (a not in df.columns.tolist())]
df_categ = df[categorical_features]
df_categ.head()
df_not_num = df_categ.select_dtypes(include = ['O'])
print('There is {} non numerical features including:\n{}'.format(len(df_not_num.columns), df_not_num.columns.tolist()))
plt.figure(figsize = (11, 15))
ax = sns.boxplot(x='Countofsellerbrands', y='HeroProduct1', data=df_categ)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)
plt.figure(figsize = (12, 16))
ax = sns.boxplot(x='Countofsellerbrands', y='HeroProduct2', data=df_categ)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)

plt.figure(figsize = (13, 15))
ax = sns.boxplot(x='Max%ofnegativesellerratingslast90days', y='HeroProduct1', data=df_categ)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)
plt.figure(figsize = (13, 16))
ax = sns.boxplot(x='Max%ofnegativesellerratingslast90days', y='HeroProduct2', data=df_categ)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)
fig, axes = plt.subplots(round(len(df_not_num.columns) / 3), 3, figsize=(12, 30))
for i, ax in enumerate(fig.axes):
    if i < len(df_not_num.columns):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        sns.countplot(x=df_not_num.columns[i], alpha=0.7, data=df_not_num, ax=ax)
fig.tight_layout()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    