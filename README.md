# MLGCN-DP for Story Ending Generation

##### Code for paper "Story Ending Generation with Multi-Level Graph Convolutional Networks over Dependency Trees", AAAI 2021.

## Prerequisites

- Python 3.6
- PyTorch == 1.9.0

## Quick start:

- Dataset

All dataset under the directory of `data`, include [ROCStories corpus](http://cs.rochester.edu/nlp/rocstories/) (`data/test`, `data/train`, `data/val`) and the dependency parsing relations (`data/DP-data.zip`, unzip it to use). We utilize the glove embedding, please download the *glove.6b.300d.txt* and put it in directory of `data/embedding`.

- Data preprocess

Run following command:

1. `python data_preprocess.py`
2. `python embed_vocab.py`

Then we will get three files

1. `data/final_gcn_data.pt`
2. `data/embedding/embedding_enc.pt`
3. `data/embedding/embedding_dec.pt`

- Training

Run command:

`python train.py -gpu 0 `

- Inference

Run command:

`python generate_story.py -gpu 0`

Then the output file will be save in directory of `story_generation/`

## PS

When calculate the BLEU, please **uniform capitals and lower case letters** of *ref* and *hyp*

