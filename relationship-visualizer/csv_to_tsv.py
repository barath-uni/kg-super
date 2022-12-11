# Simple util to clean up the csv to tsv requirements(1. Starting with KG-BERT)

import pandas as pd

output_dir = "/home/barath/codespace/kg-super/kg-super-engine/output/fb15k237/tfidvectorizer"
train_csv = f"{output_dir}/train.csv"
test_csv = f"{output_dir}/test.csv"
validation_csv = f"{output_dir}/validation.csv"
df = pd.read_csv(train_csv, escapechar='\n')
df_test = pd.read_csv(test_csv, escapechar='\n')
df_dev = pd.read_csv(validation_csv, escapechar='\n')

df.to_csv(f'{output_dir}/train.tsv', sep='\t', encoding='utf-8', index=False, header=None)
df_test.to_csv(f'{output_dir}/test.tsv', sep='\t', encoding='utf-8', index=False, header=None)
df_dev.to_csv(f'{output_dir}/dev.tsv', sep='\t', encoding='utf-8', index=False, header=None)