import numpy as np
from pykeen.datasets import get_dataset
import requests
import pandas as pd
import glob
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from functools import lru_cache
import logging

def get_data_frame(dataset_name="fb15k237"):
    dataset = get_dataset(dataset="fb15k237")
    # get a list of all the .txt files in the current directory
    txt_files = glob.glob(f'{dataset_name}/*.txt')

    dataframe = list()
    # concatenate the contents of all the .txt files into a single string
    txt = 'head\trelationship\ttail\n'
    for i, file in enumerate(txt_files):
        df = pd.read_csv(file, sep='\t')
        dataframe.append(df)
    df = pd.concat([dataframe[0], dataframe[1], dataframe[2]], axis=0)
    
    # create a DataFrame from the list of lines
    # df = pd.DataFrame(lines, columns=['head', 'relationship', 'tail'])

    # print the resulting DataFrame
    return df, dataset.entity_to_id

def cluster_relationship(df):
    unique_values = df['relationship'].unique()

    print(len(unique_values))
    # create a TfidfVectorizer to represent each value as a vector
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(unique_values)

    # create a KMeans model with K=3 clusters
    kmeans = KMeans(n_clusters=3)

    # fit the model to the vectors
    kmeans.fit(vectors)

    # predict the cluster for each vector
    clusters = kmeans.predict(vectors)
    # print the cluster for each value
    cluster_frame = [[], [], []]
    for value, cluster in zip(unique_values, clusters):
        logging.info(f'{value}: cluster {cluster}')
        cluster_frame[cluster].append(value)
    return cluster_frame


@lru_cache(maxsize=None)
def fetch_wikidata_value_for_id(id):
    try:
        word = requests.get(url=f"https://www.wikidata.org/wiki/Special:EntityData/Q{id}.json", timeout=5).json()
    except Exception as e:
        logging.error(e)
        logging.info(f"Could not get the details for {id}. Skipping this relationship")
        import sys
        sys.exit(1)
    entity = word['entities']
    if entity.get(f'Q{id}'):
        sentence_desc=entity[f'Q{id}']['descriptions']
        if sentence_desc.get('en'):
            sentence = sentence_desc['en']['value']
            return sentence
    else:
        logging.info(f"Skipping relationship = {id} because no english description is available")
        return

def get_triples_as_value(values, entity_to_id):
    rows = list()
    for row in values.iterrows():
        head = row[1]['head']
        tail=row[1]['tail']
        # Get the head and tail corresponding to this
        # Use pykeen to find the corresponding entity to entity id
        head_id=entity_to_id[head]
        tail_id=entity_to_id[tail]
        # Make a request to get the value
        head_val = fetch_wikidata_value_for_id(head_id)
        tail_val = fetch_wikidata_value_for_id(tail_id)
        if head_val == None or tail_val == None:
            logging.info("Skipping the HEAD and TAIL for the row")
            continue
        rows.append([head_val, row[1]['relationship'], tail_val])
    return rows



def get_triples_from_cluster(cluster_frame, df, entity_to_id):
    train_test_dev = [list(), list(), list()]
    # For each cluster get the triple from the dataframe
    for i, cluster in enumerate(cluster_frame):
        # For each of the cluster, get the triple for the corresponding relation
        for relation in cluster:
            train_test_dev[i].extend(get_triples_as_value(df.loc[df['relationship'] == relation], entity_to_id))

    logging.info(train_test_dev)
    # Save to CSV
    df_train = pd.DataFrame(np.asarray(train_test_dev[0]), columns = ['head','relationship','tail'])
    df_test = pd.DataFrame(np.asarray(train_test_dev[1]), columns = ['head','relationship','tail'])
    df_val = pd.DataFrame(np.asarray(train_test_dev[2]), columns = ['head','relationship','tail'])
    df_train.to_csv('train.csv', index=False)
    df_test.to_csv('test.csv', index=False)
    df_val.to_csv('validation.csv', index=False)


df, dataset_entity_id = get_data_frame("/home/barath/codespace/kg-super/kg-super-engine/fb15k237")

cluster_frame = cluster_relationship(df)

triples_from_cluster = get_triples_from_cluster(cluster_frame, df, dataset_entity_id)