## Enhancing Knowledge Graph Embedding by Composite Neighbors for Link Prediction

Source code for paper: Enhancing Knowledge Graph Embedding by Composite Neighbors for Link Prediction

### Dependencies

- Compatible with PyTorch 1.0 and Python 3.x.
- Dependencies can be installed using `requirements.txt`.

### Dataset:

- We use FB15k-237 and WN18RR dataset for knowledge graph link prediction. 
- FB15k-237 and WN18RR are included in the `data` directory. The provided code is only for link prediction task

### Training model:

- Install all the requirements from `requirements.txt.`

- Execute `utils\\buildNeiData` for extracting the composite neighbor data for dataset.

- To pretrain KGE decoder, execute `pretrain.py` and the embedding files are stored in `save\\` 

- To start training run: `main.py` loading pre-trained embedding files

