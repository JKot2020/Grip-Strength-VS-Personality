# CPSC415/515 Big Data Programming
# Project 3: Visualization and Calculations of Physical Strength & Fear-Related Personality
# Authors: William Holschuh & Jason Kotowski

###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from datetime import datetime
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

    # @TODO
    # determine and drop outliers
    
    # create and calculate means of personality traits
        # Fearfulness: hex_5, hex_29, hex_53, hex_77
        # Anxiety: hex_11, hex_35, hex_59, hex_83
        # Sentimentality: hex_17, hex_41, hex_65, hex_89
        # Emotional Dependence: hex_23, hex_47, hex_71, hex_95
    data_combined['fear_mean'] = data_combined['hex_5'].add(data_combined['hex_29']).add(data_combined['hex_53']).add(data_combined['hex_77']).div(4)
    data_combined['anx_mean'] = data_combined['hex_11'].add(data_combined['hex_35']).add(data_combined['hex_59']).add(data_combined['hex_83']).div(4)
    data_combined['sent_mean'] = data_combined['hex_17'].add(data_combined['hex_41']).add(data_combined['hex_65']).add(data_combined['hex_89']).div(4)
    data_combined['emo_mean'] = data_combined['hex_23'].add(data_combined['hex_47']).add(data_combined['hex_71']).add(data_combined['hex_95']).div(4)
    
    # clean up unused hex columns
    data_combined.drop(['hex_5', 'hex_29', 'hex_53', 'hex_77', 'hex_11', 'hex_35', 'hex_59', 'hex_83', 'hex_17', 'hex_41', 'hex_65', 'hex_89', 'hex_23', 'hex_47', 'hex_71', 'hex_95'], axis=1, inplace=True)
    
    return data_combined

# load tata
data_combined = load_data()
data_combined

###############################################################################

# @TODO
# generate graphs visualizing data

# separate male and female data points
male_data = data_combined[data_combined['female']==0].reset_index(drop=True)
female_data = data_combined[data_combined['female']==1].reset_index(drop=True)

###############################################################################

# @TODO
# generate regression model

###############################################################################

# @TODO
# generate predictive model

###############################################################################

# @TODO
# generate regression model

###############################################################################
