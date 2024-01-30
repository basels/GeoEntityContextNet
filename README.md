## Exploiting Spatial & Semantic Contexts through Embeddings for Geo-Entity Typing
### KDD 2024 Research Track (Submission #897)
__TL;DR:__ approach for embedding geo-referenced vector data, combining geometric, spatial, and semantic neighborhood contexts, for inferring geo-entity properties

__

#### Install requirements:
```commandline
pip3 install -r requirements.txt
```

### Preprocess the data
_TBD_

### Train the model
```commandline
usage: train_model.py [-h] [--train_fname TRAIN_FNAME] [--taxo_tree_fname TAXO_TREE_FNAME] [--imgs_dir IMGS_DIR] [--epochs EPOCHS]

Train Geo-Embedding model.

optional arguments:
  -h, --help            show this help message and exit
  --train_fname TRAIN_FNAME
                        Training data file name.
  --taxo_tree_fname TAXO_TREE_FNAME
                        Taxonomy tree json file name.
  --imgs_dir IMGS_DIR   Training shape images path location.
  --epochs EPOCHS       Number of epochs to train for.
  --output OUTPUT       Model output filename.
```

### Evaluate
_TBD_

#### Cite this work
_TBD_
