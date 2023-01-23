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
from sklearn.metrics import pairwise_distances
from tqdm import tqdm 
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import time
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def plot_radial_cluster(df_train, df_test, df_dev):
    relation_dev = df_dev['relationship'].unique()
    relation_test = df_test['relationship'].unique()
    relation_train = df_train['relationship'].unique()

    vectorizer = TfidfVectorizer()
    vectors_dev = vectorizer.fit_transform(relation_dev)
    vectors_train = vectorizer.fit_transform(relation_train)
    vectors_test = vectorizer.fit_transform(relation_test)

    pca = PCA(2)
    #Transform the data
    df_train = pca.fit_transform(vectors_train.todense())
    df_test = pca.fit_transform(vectors_test.todense())
    df_dev = pca.fit_transform(vectors_dev.todense())

    plt.scatter(df_train[:,0], df_train[:,1], label='train')
    plt.scatter(df_dev[:,0], df_dev[:,1], label='dev')
    plt.scatter(df_test[:,0], df_test[:,1], label='test')

    plt.legend()
    plt.title("Radial Cluster After TFID Vectorizer is applied")
    plt.savefig('k-means-tfid.png')
    plt.show()

def radially_select_clusters(df, plot=False, sim_type="tfid"):
    relations = df['relationship'].unique()
    total_triples = df.shape[0]
    print(total_triples)
    if sim_type=="tfid":
        # This is a simple change to the usual K-means, where we randomly select a cluster and radially choose items
        #  till 80% of triple is reached for train, 10% for test, 10% for validation
        # Create a TfidfVectorizer and fit it to the sentences
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(relations)
        # Use KMeans to find cluster centroids
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(vectors)
    else:
        print("SENTENCE EMBED")
        vectors, kmeans = build_sentence_vectors(df)

    if plot:
        pca = PCA(2)
        #Transform the data
        df = pca.fit_transform(vectors.todense())
        labels = kmeans.fit_predict(df)
        centroids = kmeans.cluster_centers_
        u_labels = np.unique(labels)
        #plotting the results:
        for i in u_labels:
            plt.scatter(df[labels == i , 0] , df[labels == i , 1] , label = i)
        plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
        plt.legend()
        plt.title("KMeans with TFID Vectorizer")
        plt.savefig('k-means-tfid.png')
        plt.show()
        
    already_seen_indexes = list()
    clustered_relation = list()

    def select_triples(cluster_index,split_perc=0.8):
        # Select a cluster centroid at random
        cluster_centroid = kmeans.cluster_centers_[cluster_index]

        # Select values radially outward until 80% of the sentences are selected
        selection_index = []
        prev_sentence_count = 0
        radial_distance = 0.1
        triples_list = list()
        for i, (relation, vector) in enumerate(zip(relations, vectors)):
            # We skip if the relation and vector has already been added to one of the splits (This is definitely a naive way of doing this split.
            # But there is no other way to manipulate the indexes without running into axis errors)
            if i in already_seen_indexes:
                continue
            print("CHECK FOR SEEING IF THE TOTAL TRIPLES IS GREATER THAN", split_perc*total_triples)
            if len(triples_list) > split_perc*total_triples:
                print("CURRENT LENGTH OF THE TRIPLES LIST", len(triples_list))
                print("IT IS GREATER THAN TRIPLES REQUIRED", split_perc*total_triples)
                break
            
            # Compute the distance between the sentence vector and the cluster centroid
            distance = pairwise_distances(vector.reshape(1,-1), cluster_centroid.reshape(1,-1))
            
            # If the distance is within the desired range, add the sentence to the selected sentences
            if distance <= radial_distance:
                selection_index.append(i)
                triples_list.extend(get_triples_as_value(df.loc[df['relationship'] == relation]))
                if relation in clustered_relation:
                    print(f"NOTEEEEEEEEEE... Relation = {relation} already seen.")
                else:
                    clustered_relation.append(relation)
            # Check if previous sentence count is same as the current sentence count if it is then increase the distance and continue
            if prev_sentence_count == len(selection_index):
                radial_distance = radial_distance+0.1
            
            # Reset the prev_sentence count to length of the sentences
            prev_sentence_count = len(selection_index)
        already_seen_indexes.extend(selection_index)
        return triples_list
    
    train_triples = select_triples(0)
    # Remove the selected relations from the all relations list and continue with the next centroid

    dev_triples = select_triples(1, 0.1)
    # Remove the selected relations from the all relations list and continue with the next centroid

    test_triples = list()
    # Assign Remaining triples to test
    for index, value in enumerate(relations):
        if index in already_seen_indexes:
            continue
        test_triples.extend(get_triples_as_value(df.loc[df['relationship'] == value]))
    # Save the train, test, dev to a txt file
    df_train = pd.DataFrame(np.asarray(train_triples), columns = ['head','relationship','tail'])
    df_val = pd.DataFrame(np.asarray(dev_triples), columns = ['head','relationship','tail'])
    df_test = pd.DataFrame(np.asarray(test_triples), columns = ['head','relationship','tail'])
    df_train.to_csv(f'train.csv', index=False)
    df_test.to_csv(f'test.csv', index=False)
    df_val.to_csv(f'validation.csv', index=False)
    

    logging.info("-----------------------------")
    logging.info("UNIQUE RELATIONS")
    logging.info(len(relations))
    logging.info("CLUSTER RELATIONS")
    logging.info(len(clustered_relation))
    # logging.info some statistics for further use
    logging.info("----------------------------")
    logging.info(f"Number of Triples in Train = {df_train.shape[0]}")
    logging.info(f"Number of Unique Entities(HEAD, TAIL) in Train = {len(np.unique(df_train[['head', 'tail']].values))}")
    logging.info(f"Number of Unique Relationship in Train = {len(np.unique(df_train[['relationship']].values))}")
    logging.info(f"Number of Triples in Validation = {df_val.shape[0]}")
    logging.info(f"Number of Unique Entities(HEAD, TAIL) in Validation = {len(np.unique(df_val[['head', 'tail']].values))}")
    logging.info(f"Number of Unique Relationship in Test = {len(np.unique(df_val[['relationship']].values))}")
    logging.info(f"Number of Triples in Test = {df_test.shape[0]}")
    logging.info(f"Number of Unique Entities(HEAD, TAIL) in Test = {len(np.unique(df_test[['head', 'tail']].values))}")
    logging.info(f"Number of Unique Relationship in Test = {len(np.unique(df_test[['relationship']].values))}")
    logging.info("----------------------------")

def download_nell_dataset():
    base_url = "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_{0}_ind/{1}.txt"
    # Download the dataset from each version
    versions = ['v1', 'v2', 'v3', 'v4']
    splits = ['train', 'test', 'valid']
    version_data = list()
    for split in splits:
        for version in versions:
            # Download with a request command and store to a file
            file_download = requests.get(base_url.format(version, split)).text
            with open("response.txt", "w") as f:
                f.write(file_download)
            pd_val = pd.read_csv("response.txt", sep='\t', names=['head', 'relationship', 'tail'])
            # Concatenate the pd
            version_data.append(pd_val)
    # Concatenate the whole thing as a pandas dataframe and return
    df = pd.concat([data for data in version_data], axis=0)
    return df

def get_wikidata5m_relationship_description():
    dataset = get_dataset(dataset="wikidata5m")
    relationship_dict = dict()
    # Use pykeen to get the relationship ids
    for value in dataset.relation_to_id.keys():
        logging.info(f"LOADing for {value}")
        try:
            word = requests.get(url=f"https://www.wikidata.org/wiki/Special:EntityData/{value}.json").json()
        except Exception as e:
            logging.info(f"Could not get the details for {value}. Skipping this relationship")
            continue
        # Get the english description from the json response
        sentence_desc=word['entities'][f'{value}']['descriptions']
        if sentence_desc.get('en'):
            sentence = sentence_desc['en']['value']
        else:
            logging.info(f"Skipping relationship = {value} because no english description is available")
            continue
        # Split and add it to the array
        relationship_dict[value] = sentence.encode('utf-8')
#         Slow down the request because of rate limiter
        time.sleep(0.5)
    # Fetch the description for the ids
    return relationship_dict

def get_data_frame(data_path, dataset_name="fb15k237"):
    if dataset_name != "nell995":
        dataset = get_dataset(dataset=dataset_name)
        entity_to_id = dataset.entity_to_id
        # get a list of all the .txt files in the current directory
        txt_files = glob.glob(f'{data_path}/*.txt')
        dataframe = list()
        print(data_path)
        # concatenate the contents of all the .txt files into a single string
        txt = 'head\trelationship\ttail\n'
        for i, file in enumerate(txt_files):
            df = pd.read_csv(file, sep='\t', names=['head', 'relationship', 'tail'])
            print("PANDA FRAME")
            print(df)
            dataframe.append(df)
        print(dataframe)
        df = pd.concat([dataframe[0], dataframe[1], dataframe[2]], axis=0)
        # Run a fetch to get all the description for given relationship and store in a dict
        if dataset_name == "wikidata5m":
            relationship_dict = get_wikidata5m_relationship_description()
            # Save this dict for further processing
            try:
                with open('somethingstore.json', 'w') as f:
                    f.write(str(relationship_dict).encode('utf8'))
            except Exception as e:
                logging.info("There was an exception while saving the relationship dict. Possibly because of encoding", e)
            for rel_id in relationship_dict:
                relation_desc = relationship_dict[rel_id]
                df['relationship'] = np.where(df['relationship'] == rel_id, relation_desc, df['relationship'])
    else:
        # If nell995
        df = download_nell_dataset()
        # Head, relationship, tail all have descriptions mostly, do not need additional processing at this stage
        entity_to_id = ""
    return df, entity_to_id

def build_sentence_vectors(df):
    values = df['relationship'].unique()
    corpus_embeddings = embedder.encode(values)
    clustering_model = KMeans(n_clusters=3)
    clustering_model.fit(corpus_embeddings)
    return corpus_embeddings, clustering_model

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
    # logging.info the cluster for each value
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

def get_triples_as_value(values, entity_to_id=None):
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
    # logging.info some statistics for further use
    logging.info("----------------------------")
    logging.info(f"Number of Triples in Train = {df_train.shape[0]}")
    logging.info(f"Number of Unique Entities(HEAD, TAIL) in Train = {len(np.unique(df_train[['head', 'tail']].values))}")
    logging.info(f"Number of Unique Relationship in Train = {len(np.unique(df_train[['relationship']].values))}")
    logging.info(f"Number of Triples in Test = {df_test.shape[0]}")
    logging.info(f"Number of Unique Entities(HEAD, TAIL) in Test = {len(np.unique(df_test[['head', 'tail']].values))}")
    logging.info(f"Number of Unique Relationship in Train = {len(np.unique(df_test[['relationship']].values))}")
    logging.info(f"Number of Triples in Validation = {df_val.shape[0]}")
    logging.info(f"Number of Unique Entities(HEAD, TAIL) in Validation = {len(np.unique(df_val[['head', 'tail']].values))}")
    logging.info(f"Number of Unique Relationship in Train = {len(np.unique(df_val[['relationship']].values))}")
    logging.info("----------------------------")

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = '/home/barath/codespace/kg-super/kg-super-engine/fb15k237', help = 'path to output directory')
    parser.add_argument('--data', type = str, default='fb15k237', help ='pykeen dataset name for entity and relationship id lookup')
    parser.add_argument('--output_dir', type = str, default='/home/barath/kg-super-engine/kg-super-engine/output/', help ='Number of processes')
    parser.add_argument('--cluster_type', type = str, default='tfidvectorizer', help ='Vectorizing Algorithm')
    return parser



if __name__ == "__main__":

    args = get_arg_parser().parse_args()
    df, dataset_entity_id = get_data_frame(args.data_path, args.data)
    if args.cluster_type == "tfidvectorizer":
        cluster_frame = tfid_cluster_relationship(df)
    elif args.cluster_type == "radial_cluster":
        radially_select_clusters(df, sim_type="tfid")
    else:
        cluster_frame = sentence_embedding_cluster(df)
    # get_triples_from_cluster(cluster_frame, df, dataset_entity_id, args.output_dir)
    # plt.figure(figsize=(10, 10))
    # for i in range(len(cluster_frame)):
    #     plt.scatter(cluster_frame[i, 0], cluster_frame[i, 1])
    #     plt.annotate('sentence ' + str(i), (cluster_frame[i, 0], cluster_frame[i, 1]))
    # plt.title('2D PCA projection of embedded sentences from BERT')
    # plt.show()