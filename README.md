# kg-super-engine
A knowledge graph model that can genaralize to unseen relationships without any training. Makes use of RMPI graph network along with BERT to learn embeddings for relationships and attempts to generalize from just training on the KG once

### 1. Download the data

We will make use of 2 datasets `Nell995` and `FB15K-237`. 

Move the folder `kg-super-engine/fb15k237` to required location to process it. This has the `train`, `valid`, `test` splits that are needed to run the experiments

To download Nell995, run `python relationship-visualiser/data-preprocessing.py --data Nell995 --output_dir <dir_name>`. This downloads the graph folder for processing

To download other datasets for visualization experiments, 
Download the required compressed datasets into the `data` folder:

| Download link                                                | Size (compressed) |
| ------------------------------------------------------------ | ----------------- |
| [UMLS](https://surfdrive.surf.nl/files/index.php/s/NvuKQuBetmOUe1b/download) (small graph for tests) | 121 KB            |
| [WN18RR](https://surfdrive.surf.nl/files/index.php/s/N1c8VRH0I6jTJuN/download) | 6.6 MB            |
| [FB15k-237](https://surfdrive.surf.nl/files/index.php/s/rGqLTDXRFLPJYg7/download) | 21 MB             |
| [Wikidata5M](https://surfdrive.surf.nl/files/index.php/s/TEE96zweMxsoGmR/download) | 1.4 GB            |
| [GloVe embeddings](https://surfdrive.surf.nl/files/index.php/s/zAHCIBc6PAb3NXi/download) | 423 MB            |
| [DBpedia-Entity](https://surfdrive.surf.nl/files/index.php/s/BOD7SoDTchVO9ed/download) | 1.3 GB            |

Then use `tar` to extract the files, e.g.

```sh
tar -xzvf WN18RR.tar.gz
```

### 2. Generate New datasets

To start generating new datasets, use the following scripts

For *NELL995*

`sbatch jobs/dataset/nell995_dataset.sh`

For *Wikidata5m*

`sbatch jobs/dataset/wikidata5m.sh`

For *FB15K-237*

Replace the data with 'fb15k-237' data folder path and run the same script as above.

This create 2 folders, `tfidfvectorizer` and `sentence`.

Example folders are commited and present in `kg-super-engine/output/<dataset-name>/<vectorizer>`

To generate tfidfvectorizer and sentence embedding based splits for other datasets, follow the same procedure.

#### Radial Clustering

- The radial clustering runs via the script under `relationship-visualizer/data_preprocessing.py`. Use the `cluster-type=radial_cluster` to obtain a datasplit with the radial cluster. Note the vectorizer for the 'radial-cluster' can be either TFIDF or SentenceBERT. Choose the parameters carefully to obtain the correct splits. Since the naming is not very straight-forward, this will require minimal effort to organize the output in the correct folder 

#### View stats and other information about data

At the moment, we need to manually edit the script in `relationship-visualizer/view-relationship.py` to obtain stats and other graphs for the new datasets.

You can generate

* Graph level information
* Sim-matrix between relationship of the graph
* Cardinality related statistics 

### 3. Experimental setup


#### KG-Bert Relation Prediction


#### RMPI Relation Prediction


#### Sim-PATH Relation Prediction

