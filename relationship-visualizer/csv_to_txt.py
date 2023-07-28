
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

centroid = "random_centroid_1"
output_dir = f"/home/barath/codespace/kg-super/kg-super-engine/output/nell995/sentence"
train_csv = f"{output_dir}/train.csv"
test_csv = f"{output_dir}/test.csv"
validation_csv = f"{output_dir}/validation.csv"
df = pd.read_csv(train_csv, escapechar='\n')
df_test = pd.read_csv(test_csv, escapechar='\n')
df_dev = pd.read_csv(validation_csv, escapechar='\n')

unique_ent_train = set(df['head'].unique().tolist()+df['tail'].unique().tolist())
unique_ent_dev = set(df_dev['head'].unique().tolist()+df_dev['tail'].unique().tolist())-unique_ent_train
unique_ent_test = set(df_test['head'].unique().tolist()+df_test['tail'].unique().tolist())-unique_ent_dev

print(f"UNIQUE ENTITY TEN = {len(unique_ent_train)}")
print(f"UNIQUE REL TRAIN = {len(df['relationship'].unique())}")
print(f"UNIQUE ENTITY DEV = {len(unique_ent_dev)}")
print(f"UNIQUE REL DEV = {len(df_dev['relationship'].unique())}")
print(f"UNIQUE ENTITY TEST = {len(unique_ent_test)}")
print(f"UNIQUE REL TEST = {len(df_test['relationship'].unique())}")


# df.to_csv(f"{output_dir}/train.txt", sep='\t', encoding='utf-8', index=False, header=None)

# df_dev.to_csv(f"{output_dir}/valid.txt", sep='\t', encoding='utf-8', index=False, header=None)

# df_test.to_csv(f"{output_dir}/test.txt", sep='\t', encoding='utf-8', index=False, header=None)
