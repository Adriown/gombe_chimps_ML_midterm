#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:16:16 2018

@author: mead
"""

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt

###############################################################
###                                                         ###
###    PRE-PROCESSING, FORMATTING, AND PARAMETER SETTING    ###
###                                                         ###
###############################################################
# Read in the data
df = pd.read_csv('/Users/mead/Downloads/gombe_460.csv')
# There is one duplicate entry -- chimp O198 with rater A was done twice in a 3-day period. Remove the older score.
df = df.drop_duplicates(['chimpcode', 'ratercode'])
# Keep the primary key
df_key = df[['chimpcode', 'ratercode']]
# Find unique raters
rater_list = np.unique(df_key['ratercode'])
# The data processing step to find the complete pairwise list of chimp-grader-grader groupings
count_cutoff = 20
# There are not many missing values in the columns we are interested (at most 4 in one column)
df.loc[:,'dom.raw':'innov.raw'].isnull().sum()
# The best way to deal with these missing values in not to impute, but rather to ignore 
# them during the outcome analysis step when comparing the scores between raters for that chimp

len(df['ratercode'].unique())
len(df['chimpcode'].unique())
df.hist(column = ['dom.raw', 'sol.raw', 'impl.raw', 'symp.raw', 
                 'stbl.raw', 'invt.raw', 'depd.raw', 'soc.raw', 'thotl.raw', 'help.raw', 
                 'exct.raw', 'inqs.raw', 'decs.raw', 'indv.raw', 'reckl.raw', 'sens.raw', 
                 'unem.raw', 'cur.raw', 'vuln.raw', 'actv.raw', 'pred.raw', 'conv.raw', 'cool.raw', 'innov.raw'], figsize = (25, 16), bins= 7)

###############################################################
###                                                         ###
###             CHIMP-RATER COLLECTION CREATION             ###
###                                                         ###
###############################################################
rater_pairs = DataFrame()
count = 0
for rater1 in rater_list:
    amended_list = rater_list[np.where(rater_list > rater1)]
    for rater2 in amended_list:
        rater1_chimps = df_key[df_key['ratercode'] == rater1]
        rater1_chimps = rater1_chimps.rename(columns = {'ratercode' : 'ratercode_1'})
        rater2_chimps = df_key[df_key['ratercode'] == rater2]
        rater2_chimps = rater2_chimps.rename(columns = {'ratercode' : 'ratercode_2'})
        rater_collection = rater1_chimps.merge(rater2_chimps, on = 'chimpcode')
        if len(rater_collection) >= count_cutoff:
            print("The " + rater1 + " and " + rater2 + " pair have " + str(len(rater_collection)) + ' chimps in common.')
            rater_pairs = rater_pairs.append(rater_collection)
            count += 1
print(str(count) + ' grader pairs have at least ' + str(count_cutoff) + ' chimps in common.')        

# Do the work for triples
rater_triples = DataFrame()
count = 0
for rater1 in rater_list:
    rater2_list = rater_list[np.where(rater_list > rater1)]
    for rater2 in rater2_list:
        rater3_list = rater2_list[np.where(rater2_list > rater2)]
        for rater3 in rater3_list:
            rater1_chimps = df_key[df_key['ratercode'] == rater1]
            rater1_chimps = rater1_chimps.rename(columns = {'ratercode' : 'ratercode_1'})
            rater2_chimps = df_key[df_key['ratercode'] == rater2]
            rater2_chimps = rater2_chimps.rename(columns = {'ratercode' : 'ratercode_2'})
            rater3_chimps = df_key[df_key['ratercode'] == rater3]
            rater3_chimps = rater3_chimps.rename(columns = {'ratercode' : 'ratercode_3'})
            rater_collection = rater1_chimps.merge(rater2_chimps, on = 'chimpcode')
            rater_collection = rater_collection.merge(rater3_chimps, on = 'chimpcode')
            if len(rater_collection) >= count_cutoff:
                print("The " + rater1 + ", " + rater2 + ", and " + rater3 + " triple have " + str(len(rater_collection)) + ' chimps in common.')
                rater_triples = rater_triples.append(rater_collection)
                count += 1
print(str(count) + ' grader triples have at least ' + str(count_cutoff) + ' chimps in common.')        

# Now for groupings of 4
rater_quads = DataFrame()
count = 0
for rater1 in rater_list:
    rater2_list = rater_list[np.where(rater_list > rater1)]
    for rater2 in rater2_list:
        rater3_list = rater2_list[np.where(rater2_list > rater2)]
        for rater3 in rater3_list:
            rater4_list = rater3_list[np.where(rater3_list > rater3)]
            for rater4 in rater4_list:
                rater1_chimps = df_key[df_key['ratercode'] == rater1]
                rater1_chimps = rater1_chimps.rename(columns = {'ratercode' : 'ratercode_1'})
                rater2_chimps = df_key[df_key['ratercode'] == rater2]
                rater2_chimps = rater2_chimps.rename(columns = {'ratercode' : 'ratercode_2'})
                rater3_chimps = df_key[df_key['ratercode'] == rater3]
                rater3_chimps = rater3_chimps.rename(columns = {'ratercode' : 'ratercode_3'})
                rater4_chimps = df_key[df_key['ratercode'] == rater4]
                rater4_chimps = rater4_chimps.rename(columns = {'ratercode' : 'ratercode_4'})
                rater_collection = rater1_chimps.merge(rater2_chimps, on = 'chimpcode')
                rater_collection = rater_collection.merge(rater3_chimps, on = 'chimpcode')
                rater_collection = rater_collection.merge(rater4_chimps, on = 'chimpcode')
                if len(rater_collection) >= count_cutoff:
                    print("The " + rater1 + ", " + rater2 + ", " + rater3 + ", and " + rater4 + " quad have " + str(len(rater_collection)) + ' chimps in common.')
                    rater_quads = rater_quads.append(rater_collection)
                    count += 1
print(str(count) + ' grader quads have at least ' + str(count_cutoff) + ' chimps in common.')        


# And lastly try 5 <-- clear at this point that there is also a 6-combo: A,E,M,N,Q,W
rater_quints = DataFrame()
count = 0
for rater1 in rater_list:
    rater2_list = rater_list[np.where(rater_list > rater1)]
    for rater2 in rater2_list:
        rater3_list = rater2_list[np.where(rater2_list > rater2)]
        for rater3 in rater3_list:
            rater4_list = rater3_list[np.where(rater3_list > rater3)]
            for rater4 in rater4_list:
                rater5_list = rater4_list[np.where(rater4_list > rater4)]
                for rater5 in rater5_list:
                    rater1_chimps = df_key[df_key['ratercode'] == rater1]
                    rater1_chimps = rater1_chimps.rename(columns = {'ratercode' : 'ratercode_1'})
                    rater2_chimps = df_key[df_key['ratercode'] == rater2]
                    rater2_chimps = rater2_chimps.rename(columns = {'ratercode' : 'ratercode_2'})
                    rater3_chimps = df_key[df_key['ratercode'] == rater3]
                    rater3_chimps = rater3_chimps.rename(columns = {'ratercode' : 'ratercode_3'})
                    rater4_chimps = df_key[df_key['ratercode'] == rater4]
                    rater4_chimps = rater4_chimps.rename(columns = {'ratercode' : 'ratercode_4'})
                    rater5_chimps = df_key[df_key['ratercode'] == rater5]
                    rater5_chimps = rater5_chimps.rename(columns = {'ratercode' : 'ratercode_5'})
                    rater_collection = rater1_chimps.merge(rater2_chimps, on = 'chimpcode')
                    rater_collection = rater_collection.merge(rater3_chimps, on = 'chimpcode')
                    rater_collection = rater_collection.merge(rater4_chimps, on = 'chimpcode')
                    rater_collection = rater_collection.merge(rater5_chimps, on = 'chimpcode')
                    if len(rater_collection) >= count_cutoff:
                        print("The " + rater1 + ", " + rater2 + ", " + rater3 + ", " + rater4 + ", and " + rater5 + " quint have " + str(len(rater_collection)) + ' chimps in common.')
                        rater_quints = rater_quints.append(rater_collection)
                        count += 1
print(str(count) + ' grader quints have at least ' + str(count_cutoff) + ' chimps in common.')        

# I just wrote this to make it go even faster. I know the result already so it's not too difficult to alter a small amount
rater_sexts = list(rater_quints.groupby(['ratercode_1', 'ratercode_2', 'ratercode_3', 'ratercode_4', 'ratercode_5']))[1][1]
rater_sexts['ratercode_6'] = 'Q'




###############################################################
###                                                         ###
###          FUNCTION CREATION (TO HANDLE ANALYSIS)         ###
###                                                         ###
###############################################################

def funcMergeRaters(df, rater_groups, group_size, columns_of_interest = ['chimpcode', 'ratercode', 'dom.raw', 'sol.raw', 'impl.raw', 'symp.raw', 
                                                                         'stbl.raw', 'invt.raw', 'depd.raw', 'soc.raw', 'thotl.raw', 'help.raw', 
                                                                         'exct.raw', 'inqs.raw', 'decs.raw', 'indv.raw', 'reckl.raw', 'sens.raw', 
                                                                         'unem.raw', 'cur.raw', 'vuln.raw', 'actv.raw', 'pred.raw', 'conv.raw', 'cool.raw', 'innov.raw']):
    """
    This function takes as input 'df', which is the raw chimp information as a DataFrame, 
    'rater_groups', a DataFrame with the unique chimpcode-raters combinations,
    'group_size' (> 1), an integer number of the simultaneous raters we want to consider at a 
    time (pair, triple, etc), and 'columns_of_interest', which is a list of the columns 
    we want to compare the scores for (must include 'chimpcode'.
    
    The output is the formatted DataFrame with each chimp-group rating information.
    """
    # Pull out just the columns we want to compare between raters
    raters = [df.loc[:, columns_of_interest] for i in range(1, group_size + 1)]
    # Rename them so that the columns of different raters are differentiable (_1, _2, etc)
    raters = [rater.rename(columns = lambda x: x + '_' + str(group_num + 1)) for group_num, rater in enumerate(raters)]
    # Keep the chimpcode column consistent though
    raters = [rater.rename(columns = {'chimpcode_' + str(group_num + 1) : 'chimpcode'}) for group_num, rater in enumerate(raters)]
    # Go through now and merge the raw information in the rater-chimp combos that have already been identified
    for group_num, rater in enumerate(raters):
        if group_num == 0: # First time through
            rater_tot = rater_groups.merge(rater, on = ['chimpcode', 'ratercode_' + str(group_num + 1)])
        else:
            rater_tot = rater_tot.merge(rater, on = ['chimpcode', 'ratercode_' + str(group_num + 1)])
    return rater_tot

def funcAllBayesFactor(rater_tot, group_size = 2, num_cats = 7, columns_of_interest = ['dom.raw', 'sol.raw', 'impl.raw', 'symp.raw', 
                                             'stbl.raw', 'invt.raw', 'depd.raw', 'soc.raw', 'thotl.raw', 'help.raw', 
                                             'exct.raw', 'inqs.raw', 'decs.raw', 'indv.raw', 'reckl.raw', 'sens.raw', 
                                             'unem.raw', 'cur.raw', 'vuln.raw', 'actv.raw', 'pred.raw', 'conv.raw', 'cool.raw', 'innov.raw']):
    """
    This function looks intimidating but isn't. Basically, you pass the entire merged rater-group-chimp
    DataFrame as 'rater_tot', the pair/triple/etc. value as 'group_size', the number of possible categories
    as 'num_cats' (7 for the gombe survey), and the columns that you want to calculate Baye's factors between.
    It'll return a DataFrame with the rater-group, trait, and the corresponding Baye's Factor and 
    associated likelihoods
    """
    df_bayes_factors = DataFrame()
    ratercodes = ['ratercode_' + str(i) for i in range(1, group_size + 1)]
    for name, group in rater_tot.groupby(ratercodes):
        print(name)
        for col in columns_of_interest:
            print(col)
            cols = [col + '_' + str(i) for i in range(1, group_size + 1)]
            df_cols = group[cols]
            # Important step here: Remove the row that contains any NAs from the
            # DataFrame before it is passed to the Outcome Analysis function
            if df_cols.isnull().any().any():
                print("Up to " + str(df_cols.isnull().sum().sum()) + " rows of " + str(len(df_cols)) + " total dropped for " + str(name) + " and " + col + ".")
                if (len(df_cols) - df_cols.isnull().sum().sum() < 20):
                    print("\n\n\nWARNING!!! Possibly fewer than 20 observations used to compute the Baye's Factor!\n\n\n")
                df_cols = df_cols.dropna()
            cat_counts = funcCountCategories(df_cols, group_size, num_cats)
            bayes_factor = funcOneBayesFactor(cat_counts)
            p_same = 1 / (1 + bayes_factor)
            p_diff = bayes_factor / (1 + bayes_factor)
            df_bayes_factors = df_bayes_factors.append(DataFrame([{'raters' : name, 'trait' : col.rstrip('.raw'), 'bayes_factor' : bayes_factor, 'p(diff)' : p_diff, 'p(same)' : p_same}]))
            df_bayes_factors = df_bayes_factors[['raters', 'trait', 'bayes_factor', 'p(diff)', 'p(same)']]
    return df_bayes_factors

def funcCountCategories(vec_cols, group_nums, category_nums):
    """
    This function takes as input the 'group_nums' integer as input (relating to 2 for pair, 3 for triple, etc.).
    It also takes as input the corresponding DataFrame columns of interest that we want to find the Baye's factor
    for (eg: the two dom.raw columns for the A and E rater). And it takes the number of categories to count over.
    It transforms these inputs into the category-counts that we need for input to the Dirichlet Beta function
    """
    for i in range(group_nums):
        cat_dicts = [Series(0, index = range(1, category_nums + 1), name = 'counts') for i in range(group_nums)]
    for i, row in vec_cols.iterrows():
        for j in range(group_nums):
            cat_dicts[j][row[j]] += 1
    return cat_dicts

def funcOneBayesFactor(cat_counts):
    """
    This takes as input 'cat_counts', typically the output of funcCountCategories(), which 
    is a n-tuple (where n corresponds to 2 for pair, 3 for triple, etc), that has counts for each 
    category.
    It returns the Baye's Factor associated with these values for the p(diff)/p(same)
    """
    category_nums = len(cat_counts[0])
    unit_vector = Series(1, index = range(1, category_nums + 1), name = 'counts')
    numerList = [betaFunc(unit_vector + cat_count) for cat_count in cat_counts]
    # Go through and multiply the values together
    for i in range(1, len(numerList)):
        if i == 1:
            numer = funcPrimeFactorMultiplication(numerList[i-1], numerList[i])
        else:
            numer = funcPrimeFactorMultiplication(numer, numerList[i])
    denom = funcPrimeFactorMultiplication(betaFunc(unit_vector), betaFunc(unit_vector + sum(cat_counts)))
    finalValue = funcPrimeFactorDivision(numer, denom)
    returnThis = funcEvaluatePrimeFactor(finalValue)
    return returnThis
    
def betaFunc(cat_count):
    """
    Takes a single element of 'cat_count' (typically the output of funcCountCategories())
    Returns the Beta Function applied to this vector (Dirichlet formula)
    """
    numerList = [gammaFunc(val) for val in cat_count]
    # Go through and multiply the values together
    for i in range(1, len(numerList)):
        if i == 1:
            numer = funcPrimeFactorMultiplication(numerList[i-1], numerList[i])
        else:
            numer = funcPrimeFactorMultiplication(numer, numerList[i])
    denom = gammaFunc(sum(cat_count))
    return funcPrimeFactorDivision(numer, denom)
     
def gammaFunc(value):
    """
    factorial(n-1)
    """
    return funcFactorialPrimeFactors(value - 1)

def funcPrimeFactorMultiplication(a, b):
    """
    Allows for two numbers to be multiplied using my prime factorization technique that 
    avoids integer overflow.
    """
    return a.add(b, fill_value=0).astype('int')

def funcPrimeFactorDivision(a, b):
    """
    Allows for two numbers to be divided using my prime factorization technique that 
    avoids integer overflow.
    """
    return a.subtract(b, fill_value=0).astype('int')

def funcEvaluatePrimeFactor(finalValue):
    """
    Takes a number that's been encoded in its prime factor form as I've defined it and return to a
    typical floating point representation.
    """
    returnThis = np.prod(np.power(np.asarray(list(finalValue.index), dtype = 'float'), np.asarray(list(finalValue.values))))
    return returnThis
    
def funcFactorialPrimeFactors(n):
    """
    Will take a number, 'n', to perform the factorial of n! and reduce each of the numbers 
    that are multiplied together to produce it down to their prime factorizations 
    and then compile those together to produce the n! prime factorization
    """
    factors = []
    for i in range(2, n+1):
        factors.append(funcPrimeFactors(i))
    factors = [int(item) for sublist in factors for item in sublist]
    return Series(factors).value_counts()
    
def funcPrimeFactors(n):
    """
    Returns all the prime factors of a positive integer
    """
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n /= d
        d = d + 1
        if d*d > n:
            if n > 1: factors.append(n)
            break
    return factors

def boxplot_sorted(df, by, column, title):
    """
    Code to help with boxplot plotting. Pass the 'df' as the DataFrame of interest,
    'by' as the x-axis column, 'column' as the y-axis column, and then a 'title.
    """
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    meds = df2.median().sort_values(ascending = False)
    plot_this = df2[meds.index].boxplot(rot=90, figsize = (10, 6))
    plot_this.axhline(0.5, color='black')
    fig = plot_this.get_figure()
    fig.suptitle("")
    plot_this.set_title(title)
    plot_this.set_ylabel(column)
    plot_this.set_xlabel(by)




###############################################################
###                                                         ###
###                   WORKING EXAMPLE CODE                  ###
###                                                         ###
###############################################################
# EXAMPLE CODE:
# Can I get this to work on the in-class example?
vec_cols = DataFrame([[1, 1], [3, 3], [1, 1], [3, 2], [1, 2], [1, 3], 
                         [3, 3], [2, 3], [2, 2], [3, 3], [1, 3], [1, 2],
                         [1, 2], [1, 2], [1, 2], [1, 1], [1, 2], [1, 1],
                         [1, 3], [2, 2]])
cat_counts = funcCountCategories(vec_cols, 2, 3)
funcOneBayesFactor(cat_counts)
# Expected output is 12.87 so this works fairly well
# Can I get this to work on the HW 2 Problem?
vec_cols = DataFrame([[1, 1, 1], [3, 3, 2], [1, 1, 1], [3, 2, 1], [1, 2, 1], [1, 3, 3], 
                         [3, 3, 2], [2, 3, 2], [2, 2, 2], [3, 3, 3], [1, 3, 1], [1, 2, 2],
                         [1, 2, 1], [1, 2, 2], [1, 2, 1], [1, 1, 1], [1, 2, 2], [1, 1, 3],
                         [1, 3, 3], [2, 2, 2]])
cat_counts = funcCountCategories(vec_cols, 3, 3)
funcOneBayesFactor(cat_counts)
# Expected output is 1.38 so this works fairly well







###############################################################
###                                                         ###
###             EXPLORATORY CATEGORY ANALYSIS               ###
###                                                         ###
###############################################################

df1 = df.copy()  # TFor the 7 categories
df2 = df1.copy() # For the 3 categories
df3 = df1.copy() # For the 2 categories (4 small)
df4 = df1.copy() # For the 2 categories (4 small)
# Analysis with the 7 original categories
raters_data_7 = funcMergeRaters(df = df1, rater_groups = rater_pairs, group_size = 2)
pairs_bayes_factors_7 = funcAllBayesFactor(raters_data_7, 2, 7)   

# Reduce to 3 categories (for the sake of useful insights)
df2.loc[:,'dom.raw':'innov.raw'] = np.floor(df2.loc[:,'dom.raw':'innov.raw'] / 3) + 1
# Analysis with the revised 3 categories
raters_data_3 = funcMergeRaters(df = df2, rater_groups = rater_pairs, group_size = 2)
pairs_bayes_factors_3 = funcAllBayesFactor(raters_data_3, 2, 3)   

# Reduce to 2 categories (for the sake of useful insights)
df3.loc[:,'dom.raw':'innov.raw'] = np.floor(df3.loc[:,'dom.raw':'innov.raw'] / 4.1) + 1
# Analysis with the revised 2 categories (4 is small)
raters_data_2_small = funcMergeRaters(df = df3, rater_groups = rater_pairs, group_size = 2)
pairs_bayes_factors_2_small = funcAllBayesFactor(raters_data_2_small, 2, 2)   

# Reduce to 2 categories (for the sake of useful insights)
df4.loc[:,'dom.raw':'innov.raw'] = np.floor(df4.loc[:,'dom.raw':'innov.raw'] / 4) + 1
# Analysis with the revised 2 categories (4 is small)
raters_data_2_large = funcMergeRaters(df = df4, rater_groups = rater_pairs, group_size = 2)
pairs_bayes_factors_2_large = funcAllBayesFactor(raters_data_2_large, 2, 2)   

# Merging for plotting purposes
dummy = pairs_bayes_factors_7.set_index(['raters', 'trait']).join(pairs_bayes_factors_3.set_index(['raters', 'trait']), rsuffix='_3')
dummy = dummy.join(pairs_bayes_factors_2_small.set_index(['raters', 'trait']), rsuffix='_2_sm')
dummy = dummy.join(pairs_bayes_factors_2_large.set_index(['raters', 'trait']), lsuffix = '_7', rsuffix='_2_lg')

# The large amount of sensitivity to our categorizations is a bit troubling indeed
dummy.plot.scatter(x = 'p(diff)_2_sm', y = 'p(diff)_2_lg', alpha = .4)
dummy.plot.scatter(x = 'p(diff)_3', y = 'p(diff)_7', alpha = .4)
dummy.plot.scatter(x = 'p(diff)_2_sm', y = 'p(diff)_3', alpha = .4)


###############################################################
###                                                         ###
###                     ACTUAL ANALYSIS                     ###
###                                                         ###
###############################################################
# Read in the data
df = pd.read_csv('/Users/mead/Downloads/gombe_460.csv')
# There is one duplicate entry -- chimp O198 with rater A was done twice in a 3-day period. Remove the older score.
df = df.drop_duplicates(['chimpcode', 'ratercode'])
# Keep the primary key
df_key = df[['chimpcode', 'ratercode']]
# Find unique raters
rater_list = np.unique(df_key['ratercode'])
# The data processing step to find the complete pairwise list of chimp-grader-grader groupings
count_cutoff = 20
# There are not many missing values in the columns we are interested (at most 4 in one column)
df.loc[:,'dom.raw':'innov.raw'].isnull().sum()

# Reduce to 3 categories (for the sake of useful insights)
df.loc[:,'dom.raw':'innov.raw'] = np.floor(df.loc[:,'dom.raw':'innov.raw'] / 3) + 1
num_catg = 3

#################################
###          PAIRS            ###
#################################
# ACTUALLY RUNNING PORTION
raters_data = funcMergeRaters(df = df, rater_groups = rater_pairs, group_size = 2)
pairs_bayes_factors = funcAllBayesFactor(raters_data, 2, 7)   

# Plots performed by pairs of raters with traits on the x-axis
for group in pairs_bayes_factors['raters'].unique():
    try_this = pairs_bayes_factors[pairs_bayes_factors['raters'] == group]
    plot_this = try_this.plot.bar(x = 'trait', y = 'p(diff)', legend = False, title = str(group))
    plot_this.axhline(0.5, color='black')
# Plots performed by traits with pairs of raters on the x-axis
for group in pairs_bayes_factors['trait'].unique():
    try_this = pairs_bayes_factors[pairs_bayes_factors['trait'] == group]
    plot_this = try_this.plot.bar(x = 'raters', y = 'p(diff)', legend = False, title = str(group))
    plot_this.axhline(0.5, color='black')
# Boxplot with traits on the x-axis and the p(diff) on the y-axis across all pairs
boxplot_sorted(df = pairs_bayes_factors, 
               by = 'trait',
               column = 'p(diff)', 
               title = 'p(diff) across traits for all rater pairs with > 20 chimps in common')
# Boxplot with pairs on the x-axis and the p(diff) on the y-axis across all traits
boxplot_sorted(df = pairs_bayes_factors, 
               by = 'raters',
               column = 'p(diff)', 
               title = 'p(diff) across rater pairs with > 20 chimps in common for all traits')

# Interested in producing a plot with the rater pairs on the x-axis and p(diff) on the y-axis.
# p(diff) would be calculated across ALL traits (so rbind the appropriate cols together by rater pair)
# IT TURNS OUT THAT THIS IS ACTUALLY JUST COMPLETELY USELESS. IT DOESN'T WORK
#cols_1 = [colname for colname in raters_data.columns if '.raw_1' in colname]
#cols_2 = [colname for colname in raters_data.columns if '.raw_2' in colname]
#
#raters_data_pairs_tot = DataFrame()
#for df_group in raters_data.groupby(['ratercode_1', 'ratercode_2']):
#    df_cols_1 = df_group[1][cols_1].melt()
#    df_cols_2 = df_group[1][cols_2].melt()
#    new_df = DataFrame(columns = ['ratercode_1', 'ratercode_2'])
#    new_df['.raw_1'] = df_cols_1['value']
#    new_df['.raw_2'] = df_cols_2['value']
#    new_df['ratercode_1'] = df_group[0][0]
#    new_df['ratercode_2'] = df_group[0][1]
#    raters_data_pairs_tot = raters_data_pairs_tot.append(new_df)
#
#pairs_bayes_factors_tot = funcAllBayesFactor(raters_data_pairs_tot, 2, num_catg, columns_of_interest = ['.raw'])
#plot_this = pairs_bayes_factors_tot.plot.bar(x = 'raters', y = 'p(diff)', legend = False, title = str('All Raters across all Traits'))
#plot_this.axhline(0.5, color='black')


#################################
###         TRIPLES           ###
#################################
# ACTUALLY RUNNING PORTION
raters_data = funcMergeRaters(df = df, rater_groups = rater_triples, group_size = 3)
triples_bayes_factors = funcAllBayesFactor(raters_data, 3, num_catg)

# Plots performed by triples of raters with traits on the x-axis
for group in triples_bayes_factors['raters'].unique():
    try_this = triples_bayes_factors[triples_bayes_factors['raters'] == group]
    plot_this = try_this.plot.bar(x = 'trait', y = 'p(diff)', legend = False, title = str(group))
    plot_this.axhline(0.5, color='black')
# Plots performed by traits with triples of raters on the x-axis
for group in triples_bayes_factors['trait'].unique():
    try_this = triples_bayes_factors[triples_bayes_factors['trait'] == group]
    plot_this = try_this.plot.bar(x = 'raters', y = 'p(diff)', legend = False, title = str(group))
    plot_this.axhline(0.5, color='black')
# Boxplot with traits on the x-axis and the p(diff) on the y-axis across all pairs
boxplot_sorted(df = triples_bayes_factors, 
               by = 'trait',
               column = 'p(diff)', 
               title = 'p(diff) across traits for all rater triples with > 20 chimps in common')
# Boxplot with pairs on the x-axis and the p(diff) on the y-axis across all traits
boxplot_sorted(df = triples_bayes_factors, 
               by = 'raters',
               column = 'p(diff)', 
               title = 'p(diff) across rater triples with > 20 chimps in common for all traits')
       

#################################
###        QUADRUPLES         ###
#################################
# ACTUALLY RUNNING PORTION
raters_data = funcMergeRaters(df = df, rater_groups = rater_quads, group_size = 4)
quads_bayes_factors = funcAllBayesFactor(raters_data, 4, num_catg)

# Plots performed by quadruples of raters with traits on the x-axis
for group in quads_bayes_factors['raters'].unique():
    try_this = quads_bayes_factors[quads_bayes_factors['raters'] == group]
    plot_this = try_this.plot.bar(x = 'trait', y = 'p(diff)', legend = False, title = str(group))
    plot_this.axhline(0.5, color='black')
# Plots performed by traits with quadruples of raters on the x-axis
for group in quads_bayes_factors['trait'].unique():
    try_this = quads_bayes_factors[quads_bayes_factors['trait'] == group]
    plot_this = try_this.plot.bar(x = 'raters', y = 'p(diff)', legend = False, title = str(group))
    plot_this.axhline(0.5, color='black')
# Boxplot with traits on the x-axis and the p(diff) on the y-axis across all pairs
boxplot_sorted(df = quads_bayes_factors, 
               by = 'trait',
               column = 'p(diff)', 
               title = 'p(diff) across traits for all rater quads with > 20 chimps in common')
# Boxplot with pairs on the x-axis and the p(diff) on the y-axis across all traits
boxplot_sorted(df = quads_bayes_factors, 
               by = 'raters',
               column = 'p(diff)', 
               title = 'p(diff) across rater quads with > 20 chimps in common for all traits')


#################################
###        QUINTUPLES         ###
#################################
# ACTUALLY RUNNING PORTION
raters_data = funcMergeRaters(df = df, rater_groups = rater_quints, group_size = 5)
quints_bayes_factors = funcAllBayesFactor(raters_data, 5, num_catg)

# Plots performed by quintuples of raters with traits on the x-axis
for group in quints_bayes_factors['raters'].unique():
    try_this = quints_bayes_factors[quints_bayes_factors['raters'] == group]
    plot_this = try_this.plot.bar(x = 'trait', y = 'p(diff)', legend = False, title = str(group))
    plot_this.axhline(0.5, color='black')
# Plots performed by traits with quintuples of raters on the x-axis
for group in quints_bayes_factors['trait'].unique():
    try_this = quints_bayes_factors[quints_bayes_factors['trait'] == group]
    plot_this = try_this.plot.bar(x = 'raters', y = 'p(diff)', legend = False, title = str(group))
    plot_this.axhline(0.5, color='black')
# Boxplot with traits on the x-axis and the p(diff) on the y-axis across all pairs
boxplot_sorted(df = quints_bayes_factors, 
               by = 'trait',
               column = 'p(diff)', 
               title = 'p(diff) across traits for all rater quints with > 20 chimps in common')
# Boxplot with pairs on the x-axis and the p(diff) on the y-axis across all traits
boxplot_sorted(df = quints_bayes_factors, 
               by = 'raters',
               column = 'p(diff)', 
               title = 'p(diff) across rater quints with > 20 chimps in common for all traits')


#################################
###        SEXTUPLES          ###
#################################
# ACTUALLY RUNNING PORTION
raters_data = funcMergeRaters(df = df, rater_groups = rater_sexts, group_size = 6)
sexts_bayes_factors = funcAllBayesFactor(raters_data, 6, num_catg)

# Plots performed by sextuples of raters with traits on the x-axis
for group in sexts_bayes_factors['raters'].unique():
    try_this = sexts_bayes_factors[sexts_bayes_factors['raters'] == group]
    plot_this = try_this.plot.bar(x = 'trait', y = 'p(diff)', legend = False, title = str(group), figsize = (9, 6))
    plot_this.axhline(0.5, color='black')
    plot_this.set_ylabel('p(diff)')
## Plots performed by traits with sextuples of raters on the x-axis
#for group in sexts_bayes_factors['trait'].unique():
#    try_this = sexts_bayes_factors[sexts_bayes_factors['trait'] == group]
#    plot_this = try_this.plot.bar(x = 'raters', y = 'p(diff)', legend = False, title = str(group))
#    plot_this.axhline(0.5, color='black')