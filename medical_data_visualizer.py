import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
def add_overweight_column(df):
  df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2) > 25
  return df

df = add_overweight_column(df)

# Normalize data by making 0 always good and 1 always bad
def normalize_data(df):
  df['cholesterol'] = df['cholesterol'].map({1: 0, 2: 1, 3: 1})
  df['gluc'] = df['gluc'].map({1: 0, 2: 1, 3: 1})
  return df

df = normalize_data(df)

# Convert data into long format
def convert_to_long_format(df):
  df_long = pd.melt(df, id_vars='cardio', value_vars=['cholesterol', 'gluc', 'alco', 'active', 'smoke', 'overweight'])
  return df_long

df_long = convert_to_long_format(df)

# Create chart of value counts of categorical features
def draw_cat_plot(df_long):
  fig, axes = plt.subplots(2, 1, figsize=(10, 6))
  sns.catplot(x='variable', y='value_counts', col='cardio', hue='value', data=df_long, ax=axes[0])
  sns.catplot(x='variable', y='value_counts', col='cardio', hue='value', data=df_long, ax=axes[1])
  plt.subplots_adjust(top=0.9)
  fig.suptitle('Value Counts of Categorical Features by Cardio')
  fig.savefig('catplot.png')

draw_cat_plot(df_long)

# Clean the data
def clean_data(df):
  df = df[(df['ap_lo'] <= df['ap_hi'])]
  df = df[(df['height'] >= df['height'].quantile(0.025))]
  df = df[(df['height'] <= df['height'].quantile(0.975))]
  df = df[(df['weight'] >= df['weight'].quantile(0.025))]
  df = df[(df['weight'] <= df['weight'].quantile(0.975))]
  return df

df = clean_data(df)

# Create a correlation matrix
def create_correlation_matrix(df):
  corr = df.corr()
  return corr

corr = create_correlation_matrix(df)

# Plot the correlation matrix as a heatmap
def draw_heatmap(corr):
  mask = np.triu(np.ones_like(corr, dtype=bool))
  fig, ax = plt.subplots(figsize=(10, 10))
  sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', ax=ax)
  plt.title('Correlation Matrix')
  plt.savefig('heatmap.png')

draw_heatmap(corr)
