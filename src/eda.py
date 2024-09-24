import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    df = pd.read_csv(path)
    df.rename(columns={'headline': 'text', 'is_sarcastic': 'label'}, inplace=True)
    return df

path='/Users/rianrachmanto/miniforge3/project/sarcastic_detection/data/train.csv'
df = load_data(path)
#plot the distribution of the labels using a bar plot and a pie chart using percentage
df['label'].value_counts().plot(kind='bar')
plt.show()
df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()
