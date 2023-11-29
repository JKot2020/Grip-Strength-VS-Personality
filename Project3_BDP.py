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

# histogram to display age ranges
age_hist = sns.histplot(data=data_combined['age'], bins=11).set(title='Age Range')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# pie chart to display sex
sex_count = [164, 140]
sex_name = ['Male','Female']
plt.pie(sex_count, labels=sex_name, autopct='%1.0f%%')
plt.show()

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

# plot data and best-fit line (age)
age_corr = sns.lmplot(x="grip", y="age", data=data_combined).set(title="Correlation Grip vs Age")
plt.show()

# plot data and best-fit line (sentimentality)
sent_corr = sns.lmplot(x="grip", y="sent_mean", data=data_combined).set(title="Correlation Grip vs Sentimentality")
plt.show()

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

# covariance
grip_cov = sns.lmplot(x="grip", y="sent_mean", hue="female", data=data_combined).set(title="Correlation Grip vs Sentimentality")
plt.show()


###############################################################################

# generate anxiety predictive model

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# features and target for: Anxiety
anx_features = ['anx_mean', 'age','ethnicity','female']
target = 'grip'

# seperation of features and target
X = data_combined[anx_features]
y = data_combined[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create anx_model using the RandomForestRegressor model
anx_model = RandomForestRegressor(n_estimators=100, max_depth=2, min_samples_split=3, min_samples_leaf=2, random_state=42)

# train the anx_model
anx_model.fit(X_train, y_train)

# creating predictions
y_pred = anx_model.predict(X_test)

# calculate MSE and r2 value
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R-squared: {r2 * 100}')

# scatter plot for actual vs predicted grip strength for on anx_mean
plt.scatter(X_test['anx_mean'], y_test, label='Actual Values')
sorted_indices = X_test['anx_mean'].squeeze().argsort()
plt.scatter(X_test['anx_mean'].iloc[sorted_indices], y_pred[sorted_indices], label='Predicted Values')
plt.title('Actual vs Predicted Grip Strength (Anxiety)')
plt.xlabel('Anxiety Mean')
plt.ylabel('Grip Strength')
plt.legend()
plt.show()

###############################################################################

# Predicition Model for fear_mean

# features and target for: Fearfulness
fear_features = ['fear_mean', 'age','ethnicity','female']
target = 'grip'

# seperation of features and target
X = data_combined[fear_features]
y = data_combined[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create fear_model using the RandomForestRegressor model
fear_model = RandomForestRegressor(n_estimators=100, max_depth=2, min_samples_split=3, min_samples_leaf=2, random_state=42)

# train the fear_model
fear_model.fit(X_train, y_train)

# creating predictions
y_pred = fear_model.predict(X_test)

# calculate MSE and r2 value
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R-squared: {r2 * 100}')

# scatter plot for actual vs predicted grip strength for on fear_mean
plt.scatter(X_test['fear_mean'], y_test, label='Actual Values')
sorted_indices = X_test['fear_mean'].squeeze().argsort()
plt.scatter(X_test['fear_mean'].iloc[sorted_indices], y_pred[sorted_indices], label='Predicted Values')
plt.title('Actual vs Predicted Grip Strength (Fearfulness)')
plt.xlabel('Fearfulness Mean')
plt.ylabel('Grip Strength')
plt.legend()
plt.show()

###############################################################################

# Predicition Model for sent_mean

# features and target for: Sentimentality
sent_features = ['sent_mean', 'age','ethnicity','female']
target = 'grip'

# seperation of features and target
X = data_combined[sent_features]
y = data_combined[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create sent_model using the RandomForestRegressor model
sent_model = RandomForestRegressor(n_estimators=100, max_depth=2, min_samples_split=3, min_samples_leaf=2, random_state=42)

# train the sent_model
sent_model.fit(X_train, y_train)

# creating predictions
y_pred = sent_model.predict(X_test)

# calculate MSE and r2 value
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R-squared: {r2 * 100}')

# scatter plot for actual vs predicted grip strength for on sent_mean
plt.scatter(X_test['sent_mean'], y_test, label='Actual Values')
sorted_indices = X_test['sent_mean'].squeeze().argsort()
plt.scatter(X_test['sent_mean'].iloc[sorted_indices], y_pred[sorted_indices], label='Predicted Values')
plt.title('Actual vs Predicted Grip Strength (Sentimentality)')
plt.xlabel('Sentimentality Mean')
plt.ylabel('Grip Strength')
plt.legend()
plt.show()

###############################################################################

# Predicition Model for emo_mean

# features and target for: Emotional Dependency
emo_features = ['emo_mean', 'age','ethnicity','female']
target = 'grip'

# seperation of features and target
X = data_combined[emo_features]
y = data_combined[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create emo_model using the RandomForestRegressor model
emo_model = RandomForestRegressor(n_estimators=100, max_depth=2, min_samples_split=3, min_samples_leaf=2, random_state=42)

# train the emo_model
emo_model.fit(X_train, y_train)

# creating predictions
y_pred = emo_model.predict(X_test)

# calculate MSE and r2 value
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R-squared: {r2 * 100}')

# scatter plot for actual vs predicted grip strength for on emo_mean
plt.scatter(X_test['emo_mean'], y_test, label='Actual Values')
sorted_indices = X_test['emo_mean'].squeeze().argsort()
plt.scatter(X_test['emo_mean'].iloc[sorted_indices], y_pred[sorted_indices], label='Predicted Values')
plt.title('Actual vs Predicted Grip Strength (Emotional Dependency)')
plt.xlabel('Emotional Dependency Mean')
plt.ylabel('Grip Strength')
plt.legend()
plt.show()

###############################################################################

# Predicition Model for Combined Personal Inventory

# features and target for: Combined Personal Inventory
combined_features = ['fear_mean', 'anx_mean', 'sent_mean', 'emo_mean', 'age', 'ethnicity', 'female']
target = 'grip'

# seperation of features and target
X = data_combined[combined_features]
y = data_combined[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create combined_model using the RandomForestRegressor model
combined_model = RandomForestRegressor(n_estimators=100, max_depth=2, min_samples_split=3, min_samples_leaf=2, random_state=42)

# train the combined_model
combined_model.fit(X_train, y_train)

# creating predictions
y_pred = combined_model.predict(X_test)

# calculate MSE and r2 value
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R-squared: {r2 * 100}')

# calculate mean for all means test set
X_test['combined_mean'] = X_test[['fear_mean', 'anx_mean', 'sent_mean', 'emo_mean']].mean(axis=1)

# scatter plot for actual vs predicted grip strength for all means
plt.scatter(X_test['combined_mean'], y_test, label='Actual Values')
sorted_indices = X_test['combined_mean'].squeeze().argsort()
plt.scatter(X_test['combined_mean'].iloc[sorted_indices], y_pred[sorted_indices], label='Predicted Values')
plt.title('Actual vs Predicted Grip Strength (Combined Personal Inventory)')
plt.xlabel('Combined Personal Inventory of All Means')
plt.ylabel('Grip Strength')
plt.legend()
plt.show()