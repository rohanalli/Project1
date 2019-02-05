import os
import pandas as pd
import numpy as np

DATASET_PATH = './'
data_path = os.path.join(DATASET_PATH, 'diabetes1.csv')
dataset = pd.read_csv(data_path, header=None)
#
# dataset.columns = [
#     "Pregnancies", "Glucose", "BloodPressure",
#     "SkinThickness", "Insulin", "BMI",
#     "DiabetesPedigreeFunction", "Age", "Outcome"]

# print(dataset.head())
#
# import matplotlib.pyplot as plt
# dataset.hist(grid=True, bins=20, rwidth=0.9,
#                    color='#607c8e')
# plt.show()
print(dataset[5][0])
median_bmi = dataset['BMI'].median()
dataset['BMI'] = dataset['BMI'].replace(
    to_replace=0, value=median_bmi)
print(dataset[5][0])

median_bloodp = dataset['BloodPressure'].median()
dataset['BloodPressure'] = dataset['BloodPressure'].replace(
    to_replace=0, value=median_bloodp)

median_plglcconc = dataset['Glucose'].median()
dataset['Glucose'] = dataset['Glucose'].replace(
    to_replace=0, value=median_plglcconc)

median_skinthick = dataset['SkinThickness'].median()
dataset['SkinThickness'] = dataset['SkinThickness'].replace(
    to_replace=0, value=median_skinthick)

median_twohourserins = dataset['Insulin'].median()
dataset['Insulin'] = dataset['Insulin'].replace(
    to_replace=0, value=median_twohourserins)
