# kg-super-engine
A knowledge graph model that can genaralize to unseen relationships without any training. Makes use of RMPI graph network along with BERT to learn embeddings for relationships and attempts to generalize from just training on the KG once

### 1. Download the data

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

### Idenitifying Relationship Similarity

Overlap Coefficient - 

Jaccard Coeffecient - 

CompNet - https://web.rniapps.net/compnet/ to visualize the graph

Similarity with Textual Embeddings

Cosine-Similarity - 

L2 Distance - 

#### FB15k237

- Sentence Embedding -> Kmeans (k=3)
----------------------------
Number of Triples in Train = 82436

Number of Unique Entities(HEAD, TAIL) in Train = 7644

Number of Unique Relationship in Train = 55

Number of Triples in Test = 200600

Number of Unique Entities(HEAD, TAIL) in Test = 12969

Number of Unique Relationship in Train = 132

Number of Triples in Validation = 27080

Number of Unique Entities(HEAD, TAIL) in Validation = 5961

Number of Unique Relationship in Train = 50

  
- TFIDVectorizer -> Kmeans(K=3)
----------------------------
Number of Triples in Train = 67039

Number of Unique Entities(HEAD, TAIL) in Train = 5873

Number of Unique Relationship in Train = 38

Number of Triples in Test = 42614

Number of Unique Entities(HEAD, TAIL) in Test = 6489

Number of Unique Relationship in Train = 20

Number of Triples in Validation = 200463

Number of Unique Entities(HEAD, TAIL) in Validation = 13970

Number of Unique Relationship in Train = 179

#### Nell995

- TFIDVectorizer -> Kmeans(K=3)

----------------------------
Number of Triples in Train = 24722

Number of Unique Entities(HEAD, TAIL) in Train = 6600

Number of Unique Relationship in Train = 132

Number of Triples in Test = 3

Number of Unique Entities(HEAD, TAIL) in Test = 5

Number of Unique Relationship in Train = 1

Number of Triples in Validation = 18

Number of Unique Entities(HEAD, TAIL) in Validation = 13

Number of Unique Relationship in Train = 1

- Sentence Embedding -> Kmeans(K=3)

----------------------------
Number of Triples in Train = 13032

Number of Unique Entities(HEAD, TAIL) in Train = 4962

Number of Unique Relationship in Train = 79

Number of Triples in Test = 8515

Number of Unique Entities(HEAD, TAIL) in Test = 1336

Number of Unique Relationship in Train = 25

Number of Triples in Validation = 3196

Number of Unique Entities(HEAD, TAIL) in Validation = 1196

Number of Unique Relationship in Train = 30

