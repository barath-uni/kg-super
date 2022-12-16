# Simple util to clean up the csv to tsv requirements(1. Starting with KG-BERT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
centroid = "random_centroid_2"
output_dir = f"/home/barath/codespace/kg-super/kg-super-engine/output/fb15k237/radial_cluster/{centroid}"
train_csv = f"{output_dir}/train.csv"
test_csv = f"{output_dir}/test.csv"
validation_csv = f"{output_dir}/validation.csv"
df = pd.read_csv(train_csv, escapechar='\n')
df_test = pd.read_csv(test_csv, escapechar='\n')
df_dev = pd.read_csv(validation_csv, escapechar='\n')
print("----------------------------")
print(f"Number of Triples in Train = {df.shape[0]}")
print(f"Number of Unique Entities(HEAD, TAIL) in Train = {len(np.unique(df[['head', 'tail']].values))}")
print(f"Number of Unique Relationship in Train = {len(np.unique(df[['relationship']].values))}")
print(f"Number of Triples in Test = {df_test.shape[0]}")
print(f"Number of Unique Entities(HEAD, TAIL) in Test = {len(np.unique(df_test[['head', 'tail']].values))}")
print(f"Number of Unique Relationship in Train = {len(np.unique(df_test[['relationship']].values))}")
print(f"Number of Triples in Validation = {df_dev.shape[0]}")
print(f"Number of Unique Entities(HEAD, TAIL) in Validation = {len(np.unique(df_dev[['head', 'tail']].values))}")
print(f"Number of Unique Relationship in Train = {len(np.unique(df_dev[['relationship']].values))}")
print("----------------------------")
# df.to_csv(f'{output_dir}/train.tsv', sep='\t', encoding='utf-8', index=False, header=None)
# df_test.to_csv(f'{output_dir}/test.tsv', sep='\t', encoding='utf-8', index=False, header=None)
# df_dev.to_csv(f'{output_dir}/dev.tsv', sep='\t', encoding='utf-8', index=False, header=None)

def generate_entity_degree_plot(df):
    entity_head = set(df['head'].unique())
    entity_tail = set(df['tail'].unique())
    
    entities = list(set.union(entity_head, entity_tail))
    print(len(entities))
    degree_entity_dict = dict()
    # Count Number of times entity appears in Head (out-degree)
    # Count Number of times entity appears in Tail (in-degree)
    # Add the counts as total-degree
    # Store the value as {total-degree:[entity_id]}
    for entity in entities:
        # in-degree
        in_degree = len(df.loc[df['head'] == entity])
        out_degree = len(df.loc[df['tail'] == entity])
        total_degree = in_degree+out_degree
        if total_degree in degree_entity_dict:
            degree_entity_dict[total_degree].append(entity)
        else:
            degree_entity_dict[total_degree] = [entity]
    
    degree = list()
    number_of_entities = list()
    # After exhausting the list
    # Convert the dictionary to {total_degree:number_of_entity}
    for key in degree_entity_dict:
        degree.append(key)
        number_of_entities.append(len(degree_entity_dict[key]))
    return degree, number_of_entities


degree_train, no_train = generate_entity_degree_plot(df)
degree_dev, no_dev = generate_entity_degree_plot(df_dev)

degree_test, no_test = generate_entity_degree_plot(df_test)
# Plot these points

fig, ax = plt.subplots()
ax.scatter(degree_train, no_train, alpha=0.5, marker="s", label='Train')
ax.scatter(degree_dev, no_dev, alpha=0.6, c='g', marker="d", label='Dev')
ax.scatter(degree_test, no_test, alpha=0.6, c='r', marker="*", label='Test')
ax.set_xlabel(r'Degree', fontsize=15)
ax.set_ylabel(r'Number of entities', fontsize=15)
ax.set_title('Degree vs Number of entities')
ax.grid(True)
plt.legend(loc='upper left')
fig.tight_layout()
plt.savefig(f'{centroid}.png', dpi=fig.dpi)
plt.show()