import sys
from typing import Any, List, TextIO, Tuple, Type, Union
from data_preprocessing import get_data_frame
import numpy as np
import click
import torch
import matplotlib.pyplot as plt
import transformers as transf
from pykeen.datasets import get_dataset
from nltk.stem import PorterStemmer
from keras.utils import pad_sequences
import seaborn as sns
from scipy.spatial.distance import pdist,squareform
from sklearn.decomposition import PCA
import requests
import time
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import wordninja
from sentence_transformers import SentenceTransformer
import pandas as pd

ps = PorterStemmer()
embedder = SentenceTransformer('all-MiniLM-L6-v2')


def build_train_test_dev(array_similarity_matrix, threshold):
    set_train = set()
    set_dev = set()
    set_test = set()
    for i, row in enumerate(array_similarity_matrix):
        for j, row_val in enumerate(row):
            if row_val > threshold:
                set_train.add(i)
                if len(set_dev) < 100:
                    set_dev.add(j)
                else:
                    set_test.add(j)
    # Get the intersection
    # Clean up the train, test, dev
    return set_train, set_test, set_dev

def generate_sentence_embeddings():
    # Use SBERT to generate the embeddings
    # Visualize if possible
    # Look at PCA to project these embeddings and split the dataset into Train, Test, Dev
    return

def generate_similarity_matrix_for_nell(df, sim_type="cosine", vectori="tfidf"):
    allrelations = df['relationship'].unique()
    print(len(allrelations))
    relations = []
    for rel in allrelations:
        word = rel.replace(':',' ').replace('.', ' ').replace('/', ' ').split()
        word_list = []
        for stem in word:
            for wordn in wordninja.split(stem):
                word_list.append(wordn)
        word = " ".join(word_list)
        relations.append(word)
    # print(relations)
    # import sys
    # sys.exit(1)
    if sim_type=="jaccard":
        jaccard_sim_array = np.zeros(shape=(len(relations), len(relations)))
        for x, relation1 in enumerate(relations):
            for y, relation2 in enumerate(relations):
                jaccard_sim_array[x][y] = Jaccard_Similarity(relation1, relation2)
        plt.rc('axes', titlesize=8)     # fontsize of the axes title
        plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=2)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=2)
        plt.imshow(jaccard_sim_array, cmap='summer', interpolation='nearest')
        plt.xticks(list(range(len(relations))), relations, rotation='vertical')
        plt.yticks(list(range(len(relations))), relations, rotation='horizontal')
        plt.colorbar(label="Farthest to closest", orientation="vertical")
        plt.title("Nell995 Jaccard Similarity")
        plt.savefig("nell_995_jaccard.png", dpi=400)
        plt.show()
        return
    elif sim_type == "cosine":
        if vectori == "tfidf":
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(relations)
            vectors = vectors.todense()
            array_similarity = 1-squareform(pdist(vectors, metric=sim_type))
            print(array_similarity)
        elif vectori == "sbert":
            vectors = embedder.encode(relations)
            array_similarity = 1-squareform(pdist(vectors, metric=sim_type))

        else:
            tokenizer = transf.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = transf.DistilBertModel.from_pretrained('distilbert-base-uncased')
            input_tokens = []
            for i in relations:
                input_tokens.append(tokenizer.encode(i, add_special_tokens=True))

            input_ids = pad_sequences(input_tokens, maxlen=100, dtype="long", value=0, truncating="post", padding="post")

            def create_attention_mask(input_id):
                attention_masks = []
                for sent in input_ids:
                    att_mask = [int(token_id > 0) for token_id in sent]  # create a list of 0 and 1.
                    attention_masks.append(att_mask)  # basically attention_masks is a list of list
                return attention_masks

            input_masks = create_attention_mask(input_ids)

            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(input_masks)

            # Get all the model's parameters as a list of tuples.
            params = list(model.named_parameters())

            print('The BERT model has {:} different named parameters.\n'.format(len(params)))

            print('==== Embedding Layer ====\n')

            for p in params[0:5]:
                print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

            print('\n==== First Transformer ====\n')

            for p in params[5:21]:
                print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

            print('\n==== Output Layer ====\n')

            for p in params[-4:]:
                print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
            with torch.no_grad():
                last_hidden_states = model(input_ids, attention_mask=attention_mask)
            vectors = last_hidden_states[0][:, 0, :].detach().numpy()
            print(vectors)
            array_similarity = 1-squareform(pdist(vectors, metric=sim_type))
    svm=sns.heatmap(array_similarity)
    fig=svm.get_figure()
    fig.savefig(f"nell995_{vectori}_{sim_type}.png", dpi=400)
    pca = PCA(n_components=2)
    pca.fit(vectors)
    print(np.sum(pca.explained_variance_ratio_))
    pca_sentence_features = pca.transform(vectors)
    plt.figure(figsize=(10, 10))
    for i in range(len(pca_sentence_features)):
        plt.scatter(pca_sentence_features[i, 0], pca_sentence_features[i, 1])
        plt.annotate('sentence ' + str(i), (pca_sentence_features[i, 0], pca_sentence_features[i, 1]))
    plt.title('2D PCA projection of embedded sentences')
    plt.show()
    
def generate_relationship_sim_matrix(dataset_name, similarity_type="cosine", vectori="tfidf"):
    print("Getting dataset")
    dataset = get_dataset(dataset=dataset_name)
    relationship_array = list()
    relationship_label = list()
    if dataset_name=="Wikidata5M":
        for value in dataset.relation_to_id.keys():
            print(f"LOADing for {value}")
            try:
                word = requests.get(url=f"https://www.wikidata.org/wiki/Special:EntityData/{value}.json").json()
            except Exception as e:
                print(f"Could not get the details for {value}. Skipping this relationship")
                continue
            # Get the english description from the json response
            sentence_desc=word['entities'][f'{value}']['descriptions']
            if sentence_desc.get('en'):
                sentence = sentence_desc['en']['value']
            else:
                print(f"Skipping relationship = {value} because no english description is available")
                continue
            # Split and add it to the array
            relationship_array.append(sentence.split())
    #         Slow down the request because of rate limiter
            time.sleep(1)
        relationshio_obj = {"obj":relationship_array}
        with open("wikidata5m.json", 'w') as file:
            json.dump(relationshio_obj, file)
    
    else:
        # Check for the triple in train, dev, test and add it to the array
        all_triples = [dataset.training.label_triples(dataset.training.mapped_triples), dataset.validation.label_triples(dataset.validation.mapped_triples), dataset.testing.label_triples(dataset.testing.mapped_triples)]
        # print(dataset.training.relation_id_to_label)
        # :TODO flatten this thing
        all_entity_ids = [dataset.training.entity_to_id, dataset.validation.entity_to_id, dataset.testing.entity_to_id]
        # print(dataset.training.relation_to_id)
        # Fetch the value for the entity id by appending Q<entity-id> and querying the wikidata for the value
        # 
        for value in dataset.relation_to_id.keys():
            word = value.replace('.', ' ').replace('/', ' ').split()
            word = list(set(ps.stem(stem) for stem in word))
            relationship_array.append(" ".join(word))
            # Keep another array to track the relationship label
            relationship_label.append(value)
    if similarity_type=="jaccard":
        jaccard_sim_array = np.zeros(shape=(len(relationship_array), len(relationship_array)))
        for x, relation1 in enumerate(relationship_array):
            for y, relation2 in enumerate(relationship_array):
                jaccard_sim_array[x][y] = Jaccard_Similarity(relation1, relation2)

        sorted_index = get_pca_ordering(jaccard_sim_array)
        # send the sentence embeddings and the sorted index for further processing and storing
        # generate_csv(sorted_index, relationship_label, all_triples, all_entity_ids)
        plt.imshow(jaccard_sim_array, cmap='summer', interpolation='nearest')
        plt.xticks(list(range(len(relationship_array))), relationship_array, rotation='vertical')
        plt.yticks(list(range(len(relationship_array))), relationship_array, rotation='horizontal')
        plt.colorbar(label="Farthest to closest", orientation="vertical")
        plt.title(dataset_name)
        plt.show()

    elif similarity_type=="cosine":
        # Load pretrained model/tokenizer
        # :TODO Use FastTokenizer
        if vectori == "tfidf":
            relations = []
            print(relationship_array)
            for rel in relationship_array:
                # word = rel.replace(':',' ').replace('.', ' ').replace('/', ' ').split()
                # word_list = []
                # for stem in word:
                #     for wordn in wordninja.split(stem):
                #         word_list.append(wordn)
                word = " ".join(rel)
                relations.append(word)
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(relations)
            sentence_features = vectors.todense()
            # array_similarity = 1-squareform(pdist(vectors, metric=similarity_type))
        elif vectori == "sbert":
            sentence_features = embedder.encode(relationship_array)
            print(pdist(sentence_features, metric=similarity_type))
        else:
            tokenizer = transf.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = transf.DistilBertModel.from_pretrained('distilbert-base-uncased')
            input_tokens = []
            for i in relationship_array:
                input_tokens.append(tokenizer.encode(i, add_special_tokens=True))

            input_ids = pad_sequences(input_tokens, maxlen=100, dtype="long", value=0, truncating="post", padding="post")

            def create_attention_mask(input_id):
                attention_masks = []
                for sent in input_ids:
                    att_mask = [int(token_id > 0) for token_id in sent]  # create a list of 0 and 1.
                    attention_masks.append(att_mask)  # basically attention_masks is a list of list
                return attention_masks

            input_masks = create_attention_mask(input_ids)

            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(input_masks)

            # Get all the model's parameters as a list of tuples.
            params = list(model.named_parameters())

            print('The BERT model has {:} different named parameters.\n'.format(len(params)))

            print('==== Embedding Layer ====\n')

            for p in params[0:5]:
                print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

            print('\n==== First Transformer ====\n')

            for p in params[5:21]:
                print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

            print('\n==== Output Layer ====\n')

            for p in params[-4:]:
                print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
            with torch.no_grad():
                last_hidden_states = model(input_ids, attention_mask=attention_mask)
            sentence_features = last_hidden_states[0][:, 0, :].detach().numpy()
        print(sentence_features)
        array_similarity = 1-squareform(pdist(sentence_features, metric=similarity_type))
        # Way to fetch maybe 10-100 items that have a high cosine similarity and look at the 'description-relationship'
        train, dev, test=build_train_test_dev(array_similarity, threshold=0.25)
        # save train, dev, test as csv
        svm=sns.heatmap(array_similarity)
        fig=svm.get_figure()
        fig.savefig(f"{dataset_name}_{vectori}_{similarity_type}.png", dpi=400)
        pca = PCA(n_components=2)
        pca.fit(sentence_features)
        print(np.sum(pca.explained_variance_ratio_))
        pca_sentence_features = pca.transform(sentence_features)
        plt.figure(figsize=(10, 10))
        for i in range(len(pca_sentence_features)):
            plt.scatter(pca_sentence_features[i, 0], pca_sentence_features[i, 1])
            plt.annotate('sentence ' + str(i), (pca_sentence_features[i, 0], pca_sentence_features[i, 1]))
        plt.title('2D PCA projection of embedded sentences')
        plt.show()
        plt.savefig(f"{dataset_name}_{vectori}_pca_plot.png")
    else:
        print("This similarity type is not available")


def get_pca_ordering(embeddings):
    # For jaccard, embeddings, perform a 2-d PCA and return
    pca = PCA(n_components=1)
    pca.fit(embeddings)
    pca_sentence_features = pca.transform(embeddings)
    print(pca_sentence_features)
    plt.figure(figsize=(2,2))
    for i in range(len(pca_sentence_features)):
        plt.scatter(pca_sentence_features[i],0)
        plt.annotate('sentence ' + str(i), (pca_sentence_features[i],0))
    plt.title('2D PCA projection of embedded sentences from BERT')
    plt.show()
    return sorted(range(len(pca_sentence_features)), key=lambda k: pca_sentence_features[k])

def get_entity_name(value):
    # Query wikidata, parse and return
    return

def generate_csv(sorted_index, relationship_label, all_triples, all_entity_ids):
    # loop through sorted_index
    # For each index, get the relationship label
    # get all triples with 'relationship label' from train, validation, text
    # For corresponding train, validation, test get the respective entity id -> Query Wikidata -> add it to an array
    # Build this list and save it to a csv
    # Load csv, split it with 0.8, 0.1, 0.1 split for the new train, validation, test
    global_triples_list = list()
    for index in sorted_index:
        relationship_name = relationship_label[index]
        all_triples = list()
        for set_val in all_triples:
            for value in set_val:
                # Check if value[1] matches the relationship id, if it does, update the triples with this
                if value[1] == relationship_name:
                    # Get the entity description from the triple
                    all_triples.append([get_entity_name(all_entity_ids[value[0]]), value[1], get_entity_name(all_entity_ids[value[2]])])
        global_triples_list.extend(all_triples)
    with open("something.txt", 'w') as f:
        f.write(str(global_triples_list))

    #:TODO Logic to split it to train, validation, test

    # Values are triples, form a csv for the given name
    return

def Jaccard_Similarity(words_doc1, words_doc2):
    words_doc1 = set(words_doc1)
    words_doc2 = set(words_doc2)
    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)

    # Calculate Jaccard similarity score 
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)


@click.command()
@click.option('--dataset', required=True)
@click.option('--features', type=int, default=2, show_default=True)
@click.option('--seed', type=int, default=2, show_default=True)
@click.option('--output', type=click.File('w'))
def main(dataset, features: int, seed: int, output: TextIO):
    """Generate random literals for the Nations dataset."""
    for h, r, t in generate_literals(dataset=dataset, features=features, seed=seed):
        print(h, r, t, sep='\t', file=output)

def get_cardinality_stats(df):
    unique_relations = df['relationship'].unique()
    one_to_n=0
    n_to_one=0
    n_to_n=0
    one_to_one = 0
    for rel in unique_relations:
        # Get matching triples
        matching_triples = df[df['relationship']==rel]
        # Get the unique head, tail
        unique_head = matching_triples['head'].unique()
        unique_tail = matching_triples['tail'].unique()
        head_len = len(unique_head)
        tail_len = len(unique_tail)
        if head_len > 1 and tail_len > 1:
            n_to_n += 1
        elif (head_len == 1 and tail_len > 1):
            one_to_n += 1
        elif (head_len > 1 and tail_len == 1):
            n_to_one += 1
        else:
            one_to_one += 1
    # print("CARDINALITY STATS IS")
    return {'n_to_n':n_to_n, "1_to_n":one_to_n, "n_to_1":n_to_one, "1_to_1":one_to_one}
    # print('N to N', n_to_n)
    # print('1 to N', one_to_n)
    # print('N to 1', n_to_one)
    # print('1 to 1', one_to_one)

def generate_dataset_stats_plot( path):
    # Cardinality Stats -> For each relation in the set, get its cardinality (1-N?N-1?N-N)
    # df.relation unique(), then get the set of rows that has this relation, count unique head/tail. Add it to the corresponding bucket -> (1-N, N-1, N-N)
    train_csv = f"{path}/train.csv"
    test_csv = f"{path}/test.csv"
    validation_csv = f"{path}/validation.csv"
    df = pd.read_csv(train_csv, escapechar='\n')
    df_test = pd.read_csv(test_csv, escapechar='\n')
    df_dev = pd.read_csv(validation_csv, escapechar='\n')

    train_card = get_cardinality_stats(df)
    dev_card = get_cardinality_stats(df_dev)
    test_card = get_cardinality_stats(df_test)
    
    print(train_card)
    print(dev_card)
    print(test_card)

def get_entity_type_plot(path):
    # Type of entity, seen-rel-seen, unseen-rel-seen/seen-rel-unseen, unseen-rel-unseen
    # For dev/test only, for dev get unique entities head,tail from train, is head,tail->seen? seen-rel-seen+1, is head unseen, tail seen/tail unseen, head seen, then that bucket
    # For test, unique entity from train+dev repeat above
    train_csv = f"{path}/train.csv"
    test_csv = f"{path}/test.csv"
    validation_csv = f"{path}/validation.csv"
    df = pd.read_csv(train_csv, escapechar='\n')
    df_test = pd.read_csv(test_csv, escapechar='\n')
    df_dev = pd.read_csv(validation_csv, escapechar='\n')
    unique_head = df['head'].unique().tolist()
    unique_tail = df['tail'].unique().tolist()
    
    unique_entities = list(set(unique_head+unique_tail))
    seen_rel_seen = 0
    seen_rel_unseen = 0
    unseen_rel_unseen = 0
    for relation in df_dev['relationship'].unique():
        matching_triples = df_dev[df_dev['relationship']==relation]
        # Get entities
        # matching_head = matching_triples['head'].unique()
        # matching_tail = matching_triples['tail'].unique()
        for triples in matching_triples.iterrows():
            head = triples[1]['head']
            tail = triples[1]['tail']
            if ((head in unique_entities) and (tail not in unique_entities)) or ((head not in unique_entities) and (tail in unique_entities)):
                seen_rel_unseen += 1
            elif (head not in unique_entities) and (tail not in unique_entities):

                unseen_rel_unseen += 1
            else:
                seen_rel_seen += 1
    print("DEV Set")
    print("Seen to unseen", seen_rel_unseen)
    print("Unseen to Unseen", unseen_rel_unseen)
    print("Seen to Seen", seen_rel_seen)

    # # # # # 
    seen_rel_seen = 0
    seen_rel_unseen = 0
    unseen_rel_unseen = 0
    unique_entities = list(set(df['head'].unique().tolist()+df['tail'].unique().tolist()+df_dev['head'].unique().tolist()+df_dev['tail'].unique().tolist()))
    for relation in df_test['relationship'].unique():
        matching_triples = df_test[df_test['relationship']==relation]
        # Get entities
        # matching_head = matching_triples['head'].unique()
        # matching_tail = matching_triples['tail'].unique()
        counter = 0
        for triples in matching_triples.iterrows():
            head = triples[1]['head']
            tail = triples[1]['tail']
            if ((head in unique_entities) and (tail not in unique_entities)) or ((head not in unique_entities) and (tail in unique_entities)):
                seen_rel_unseen += 1
            elif (head not in unique_entities) and (tail not in unique_entities):
                unseen_rel_unseen += 1
                
            else:
                
                seen_rel_seen += 1

    print("TEST Set")
    print("Seen to unseen", seen_rel_unseen)
    print("Unseen to Unseen", unseen_rel_unseen)
    print("Seen to Seen", seen_rel_seen)

if __name__ == '__main__':
    # main()
    # generate_relationship_sim_matrix("FB15k237", "cosine")
    # generate_relationship_sim_matrix("WN18RR", "cosine")
    # df, dataset_entity_id = get_data_frame("", "nell995")
    centroid = "random_centroid_1"
    path = f"/home/barath/codespace/kg-super/kg-super-engine/output/fb15k237/radial_cluster_sentence/{centroid}"
    # get_entity_type_plot(path)
    generate_dataset_stats_plot(path)
    # generate_similarity_matrix_for_nell(df,"cosine", "sbert")
    # generate_relationship_sim_matrix("FB15k237")
    # generate_relationship_sim_matrix("WN18RR", vectori="sbert")
    # generate_relationship_sim_matrix("FB15k237", vectori="sbert")
    # generate_relationship_sim_matrix("FB15k237", similarity_type="jaccard")
    # Generate a split based on the similarity
    # cluster them first? or split them 80-20 and then split the 20 with 10-10 for dev and test