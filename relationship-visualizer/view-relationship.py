import sys
from typing import Any, List, TextIO, Tuple, Type, Union
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

ps = PorterStemmer()


def generate_relationship_sim_matrix(dataset_name, similarity_type="jaccard"):
    print("Getting dataset")
    dataset = get_dataset(dataset=dataset_name)
    relationship_array = list()
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
    else:
        for value in dataset.relation_to_id.keys():
            word = value.replace('.', ' ').replace('/', ' ').split()
            word = list(set(ps.stem(stem) for stem in word))
            relationship_array.append(word)
    if similarity_type=="jaccard":
        jaccard_sim_array = np.zeros(shape=(len(relationship_array), len(relationship_array)))
        for x, relation1 in enumerate(relationship_array):
            for y, relation2 in enumerate(relationship_array):
                jaccard_sim_array[x][y] = Jaccard_Similarity(relation1, relation2)

        plt.imshow(jaccard_sim_array, cmap='summer', interpolation='nearest')
        plt.xticks(list(range(len(relationship_array))), relationship_array, rotation='vertical')
        plt.yticks(list(range(len(relationship_array))), relationship_array, rotation='horizontal')
        plt.colorbar(label="Farthest to closest", orientation="vertical")
        plt.title(dataset_name)
        plt.show()
    elif similarity_type=="cosine":
        # Load pretrained model/tokenizer
        # :TODO Use FastTokenizer
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
        array_similarity = squareform(pdist(sentence_features, metric=similarity_type))
        # Way to fetch maybe 10-100 items that have a high cosine similarity and look at the 'description-relationship'
        svm=sns.heatmap(array_similarity)
        fig=svm.get_figure()
        fig.savefig(f"{dataset_name}_{similarity_type}.png", dpi=400)
        pca = PCA(n_components=10)
        pca.fit(sentence_features)
        print(np.sum(pca.explained_variance_ratio_))
        pca_sentence_features = pca.transform(sentence_features)
        plt.figure(figsize=(10, 10))
        for i in range(len(pca_sentence_features)):
            plt.scatter(pca_sentence_features[i, 0], pca_sentence_features[i, 1])
            plt.annotate('sentence ' + str(i), (pca_sentence_features[i, 0], pca_sentence_features[i, 1]))
        plt.title('2D PCA projection of embedded sentences from BERT')
        plt.show()

    else:
        print("This similarity type is not available")


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


if __name__ == '__main__':
    # main()
    # generate_relationship_sim_matrix("FB15k237", "cosine")
    # generate_relationship_sim_matrix("WN18RR", "cosine")
    generate_relationship_sim_matrix("Wikidata5M")
