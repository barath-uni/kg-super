from typing import Any, List, TextIO, Tuple, Type, Union
import numpy as np
import click
import torch
import matplotlib.pyplot as plt
import pykeen.nn
from pykeen.datasets import get_dataset

def generate_relationship_sim_matrix(dataset_name):
    dataset = get_dataset(dataset=dataset_name)

    relationship_array = list()
    for value in dataset.relation_to_id.keys():
        relationship_array.append(value.replace('.', ' ').replace('/', ' ').split())
    relationship_array=relationship_array[:20]
    jaccard_sim_array=np.zeros(shape=(len(relationship_array), len(relationship_array)))
    for x, relation1 in enumerate(relationship_array):
        for y, relation2 in enumerate(relationship_array):
            jaccard_sim_array[x][y] = Jaccard_Similarity(relation1, relation2)
    
    plt.imshow(jaccard_sim_array, cmap='summer', interpolation='nearest')
    plt.xticks(list(range(len(relationship_array))), relationship_array, rotation='vertical')
    plt.yticks(list(range(len(relationship_array))), relationship_array, rotation='horizontal')
    plt.colorbar(label="Closest to Farthest", orientation="vertical")
    plt.show()

def Jaccard_Similarity(words_doc1, words_doc2): 

    words_doc1=set(words_doc1)
    words_doc2=set(words_doc2)
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
    generate_relationship_sim_matrix("FB15k237")
