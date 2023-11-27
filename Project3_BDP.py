# CPSC415/515 Big Data Programming
# Project 3: Visualization and Calculations of Physical Strength & Fear-Related Personality
# Authors: William Holschuh & Jason Kotowski

###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
plt.rc("figure", figsize=(10, 6))
rm = np.random.default_rng(10) # 10 randome seed

# project input parameters
physical_strength_personality_file_name = "Sample_5_corrected.csv"

# parameters for simulation
num_simulations = 100
confidence = 0.95

###############################################################################

# Load survey data
# Clean, organize and generate a data frame
# Use regular expression when needed
# @return: a DataFrame with header: []

def load_data():
    # read from .csv file
    survey_data = pd.read_csv("Sample_5_corrected.csv", keep_default_na=False)
    
    # create dataframe
    data_combined = pd.DataFrame(survey_data)

    # drop outliers
    data_combined = data_combined.drop(62).reset_index(drop=True)
    
    # create and calculate means of personality traits
        # Fearfulness: hex_5, hex_29, hex_53, hex_77
        # Anxiety: hex_11, hex_35, hex_59, hex_83
        # Sentimentality: hex_17, hex_41, hex_65, hex_89
        # Emotional Dependence: hex_23, hex_47, hex_71, hex_95
    data_combined['fear_mean'] = data_combined['hex_5'].add(data_combined['hex_29']).add(data_combined['hex_53']).add(data_combined['hex_77']).div(4)
    data_combined['anx_mean'] = data_combined['hex_11'].add(data_combined['hex_35']).add(data_combined['hex_59']).add(data_combined['hex_83']).div(4)
    data_combined['sent_mean'] = data_combined['hex_17'].add(data_combined['hex_41']).add(data_combined['hex_65']).add(data_combined['hex_89']).div(4)
    data_combined['emo_mean'] = data_combined['hex_23'].add(data_combined['hex_47']).add(data_combined['hex_71']).add(data_combined['hex_95']).div(4)
    
    # clean up unused columns
    data_combined.drop(['p_id', 'hex_5', 'hex_29', 'hex_53', 'hex_77', 'hex_11', 'hex_35', 'hex_59', 'hex_83', 'hex_17', 'hex_41', 'hex_65', 'hex_89', 'hex_23', 'hex_47', 'hex_71', 'hex_95'], axis=1, inplace=True)
    
    return data_combined

# load tata
data_combined = load_data()
data_combined

###############################################################################

# @TODO
# generate graphs visualizing data

# boxplot
grip_box = sns.boxplot(x='female', y='grip',
                       data=data_combined).set(title='Grip Strength by Sex')
plt.xlabel('Sex')
plt.ylabel('Grip Strength')
plt.show(grip_box)

###############################################################################

# generate regression model

# separate male and female data points
data_male = data_combined[data_combined['female']==0].reset_index(drop=True)
data_female = data_combined[data_combined['female']==1].reset_index(drop=True)

# display OLS results (combined)
pd.options.display.max_rows = 20
import statsmodels.formula.api as smf
mod = smf.ols(formula = "grip ~ age + female + ethnicity + fear_mean + anx_mean + sent_mean + emo_mean",
              data = data_combined.dropna())
res = mod.fit()
print(res.params)
print(res.summary())

# display OLS results (male)
mod = smf.ols(formula = "grip ~ age + ethnicity + fear_mean + anx_mean + sent_mean + emo_mean",
              data = data_male.dropna())
res = mod.fit()
print(res.params)
print(res.summary())

# display OLS results (female)
mod = smf.ols(formula = "grip ~ age + ethnicity + fear_mean + anx_mean + sent_mean + emo_mean",
              data = data_female.dropna())
res = mod.fit()
print(res.params)
print(res.summary())

###############################################################################

# @TODO
# generate predictive model

###############################################################################
