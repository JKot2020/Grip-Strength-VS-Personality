# CPSC415/515 Big Data Programming
# Project 3: Visualization and Calculations of Physical Strength & Fear-Related Personality
# Authors: William Holschuh & Jason Kotowski

###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices, dmatrix
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
    data_combined = data_combined.drop(62)
    
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

# load data
data_combined = load_data()
data_combined

###############################################################################

# @TODO
# generate graphs visualizing data

# separate male and female data points
data_male = data_combined[data_combined['female']==0].reset_index(drop=True)
data_female = data_combined[data_combined['female']==1].reset_index(drop=True)

grip_box = sns.boxplot(x='ethnicity', y='grip',
                       data=data_combined).set(title='Grip Strength by Ethnicity')
plt.xlabel('Ethnicity')
plt.ylabel('Grip Strength')
plt.show()

grip_box = sns.boxplot(x='age', y='grip',
                       data=data_combined).set(title='Grip Strength by Age')
plt.xlabel('Age')
plt.ylabel('Grip Strength')
plt.show()

# boxplot
grip_box = sns.boxplot(x='female', y='grip',
                       data=data_combined).set(title='Grip Strength by Sex')
plt.xlabel('Sex (0 = male, 1 = female)')
plt.ylabel('Grip Strength')
plt.show()

# histogram of fearfulness
fear_hist = sns.histplot(data=data_combined["fear_mean"], bins=12, color='purple').set(title='Personality Rating: Fearfulness')
plt.xlabel('Feafulness Rating (Mean)')
plt.ylabel('Frequency')
plt.show()

# histogram of anxiety
anx_hist = sns.histplot(data=data_combined["anx_mean"], bins=12, color='red').set(title='Personality Rating: Anxiety')
plt.xlabel('Anxiety Rating (Mean)')
plt.ylabel('Frequency')
plt.show()

# histogram of sentimentality
sent_hist = sns.histplot(data=data_combined["sent_mean"], bins=12, color='green').set(title='Personality Rating: Sentimentality')
plt.xlabel('Sentimentality Rating (Mean)')
plt.ylabel('Frequency')
plt.show()

# histogram of emotional dependence
emo_hist = sns.histplot(data=data_combined["emo_mean"], bins=12, color='yellow').set(title='Personality Rating: Emotional Dependence')
plt.xlabel('Emotion Dependence Rating (Mean)')
plt.ylabel('Frequency')
plt.show()

###############################################################################

# generate Tukey's 5-Number summary

tukey_combined = data_combined.describe()
tukey_male = data_male.describe()
tukey_female = data_female.describe()

tukey_combined, tukey_male, tukey_female

###############################################################################

# generate regression model

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

# model prediction
y2, X_model2 = dmatrices("female ~ grip + np.log(age) + ethnicity + fear_mean + anx_mean + sent_mean + emo_mean", data_combined, return_type='dataframe')
model2 = sm.OLS(y2, X_model2)
results2 = model2.fit()
y2_pred = results2.predict(X_model2)
print(y2.shape, X_model2.shape)
print('y2 = ', y2[:10])
print('y2_pred=', y2_pred[:10])

yT = np.asarray(y2['female'], dtype=int)
yP = np.asarray(y2_pred > 0.5, dtype=int)

diff = yT - yP
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(yT, yP) * 100
print('Prediction Accuracy:', accuracy)
print('False Negatives (True=0):', np.sum(diff < 0))
print('False Positives:', np.sum(diff > 0))

# plot predicted values and true values
fig, ax = plt.subplots()
ax.plot(np.arange(len(yT)), yT - yP, "+", label="True-Predict")
plt.show()

y, X_model = dmatrices("female ~ grip + np.log(age) + ethnicity + fear_mean + anx_mean + sent_mean + emo_mean", data_combined, return_type='dataframe')
print('Total Sample Size:', len(y))

nTrain = int(len(y) * 0.6)

model_train = sm.Logit(y['female'][:nTrain], X_model[:nTrain])
results_train = model_train.fit(disp=0)
y2_pred_train = results_train.predict(X_model[nTrain:])

yT_train = y['female'].iloc[nTrain:]
yP_train = np.asarray(y2_pred_train > 0.5, dtype=int)

diff_train = yT_train - yP_train
accuracy_train = (1 - np.sum(np.abs(diff_train)) / len(diff_train)) * 100

print('Train Sample Size:', nTrain)
print('Prediction Diff:', diff_train[:10])
print('Accuracy:', accuracy_train)
print('Total Prediction:', len(diff_train))
print('False Negatives (True=0):', np.sum(diff_train < 0))
print('False Positives:', np.sum(diff_train > 0))

###############################################################################
