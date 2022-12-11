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
import argparse
from tqdm import tqdm 
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def get_data_frame(data_path, dataset_name="fb15k237"):
    dataset = get_dataset(dataset=dataset_name)
    # get a list of all the .txt files in the current directory
    txt_files = glob.glob(f'{data_path}/*.txt')

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


def sentence_embedding_cluster(df):
    values = df['relationship'].unique()
    corpus_embeddings = embedder.encode(values)
    clustering_model = KMeans(n_clusters=3)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = [[] for i in range(3)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(values[sentence_id])
    return clustered_sentences

def jaccard_sim_vectors(values):
    return

def tfid_cluster_relationship(df):
    unique_values = df['relationship'].unique()

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
        # TODO: Skipping this lookup for now. Seems to be timetaking and not properly rendering in LISA
        # Get the head and tail corresponding to this
        # Use pykeen to find the corresponding entity to entity id
        # head_id=entity_to_id[head]
        # tail_id=entity_to_id[tail]
        # # Make a request to get the value
        # head_val = fetch_wikidata_value_for_id(head_id)
        # tail_val = fetch_wikidata_value_for_id(tail_id)
        # if head_val == None or tail_val == None:
        #     logging.info("Skipping the HEAD and TAIL for the row")
        #     continue
        rows.append([head, row[1]['relationship'], tail])
    return rows



def get_triples_from_cluster(cluster_frame, df, entity_to_id, output_dir):
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
    df_train.to_csv(f'{output_dir}/train.csv', index=False)
    df_test.to_csv(f'{output_dir}/test.csv', index=False)
    df_val.to_csv(f'{output_dir}/validation.csv', index=False)
    # Print some statistics for further use
    print("----------------------------")
    print(f"Number of Triples in Train = {df_train.shape[0]}")
    print(f"Number of Unique Entities(HEAD, TAIL) in Train = {len(np.unique(df_train[['head', 'tail']].values))}")
    print(f"Number of Unique Relationship in Train = {len(np.unique(df_train[['relationship']].values))}")
    print(f"Number of Triples in Test = {df_test.shape[0]}")
    print(f"Number of Unique Entities(HEAD, TAIL) in Test = {len(np.unique(df_test[['head', 'tail']].values))}")
    print(f"Number of Unique Relationship in Train = {len(np.unique(df_test[['relationship']].values))}")
    print(f"Number of Triples in Validation = {df_val.shape[0]}")
    print(f"Number of Unique Entities(HEAD, TAIL) in Validation = {len(np.unique(df_val[['head', 'tail']].values))}")
    print(f"Number of Unique Relationship in Train = {len(np.unique(df_val[['relationship']].values))}")
    print("----------------------------")

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = '/home/barath/kg-super-engine/kg-super-engine/fb15k237', help = 'path to output directory')
    parser.add_argument('--data', type = str, default='fb15k237', help ='pykeen dataset name for entity and relationship id lookup')
    parser.add_argument('--output_dir', type = str, default='/home/barath/kg-super-engine/kg-super-engine/output/', help ='Number of processes')
    parser.add_argument('--cluster_type', type = str, default='tfidvectorizer', help ='Vectorizing Algorithm')
    return parser



if __name__ == "__main__":

    args = get_arg_parser().parse_args()
    df, dataset_entity_id = get_data_frame(args.data_path, args.data)
    # if args.cluster_type == "tfidvectorizer":
    #     cluster_frame = tfid_cluster_relationship(df)
    # else:
    #     cluster_frame = sentence_embedding_cluster(df)
    # get_triples_from_cluster(cluster_frame, df, dataset_entity_id, args.output_dir)