## Exploiting Spatial & Semantic Contexts through Embeddings for Geo-Entity Typing
### KDD 2024 Research Track (Submission #897)
__TL;DR:__ approach for embedding geo-referenced vector data, combining geometric, spatial, and semantic neighborhood contexts, for inferring geo-entity properties

__

#### Install requirements:
```commandline
pip3 install -r requirements.txt
```

### Preprocess the data
```commandline
usage: preprocess_data.py [-h] --input_osm INPUT_OSM [--output_dir OUTPUT_DIR] [--output_file OUTPUT_FILE]

Pre-process the OSM data for the training of Geo-Embedding model.

optional arguments:
  -h, --help            show this help message and exit
  --input_osm INPUT_OSM
                        OSM dump (xml) input filename.
  --output_dir OUTPUT_DIR
                        Output OSM images directory path.
  --output_file OUTPUT_FILE
                        output filename (used for the training script).
```

### Train and evaluate the model
```commandline
usage: train_model.py [-h] --train_fname TRAIN_FNAME [--taxo_tree_fname TAXO_TREE_FNAME] [--imgs_dir IMGS_DIR] [--epochs EPOCHS] [--eval_fname EVAL_FNAME] [--output OUTPUT]

Train Geo-Embedding model.

optional arguments:
  -h, --help            show this help message and exit
  --train_fname TRAIN_FNAME
                        Training data file name.
  --taxo_tree_fname TAXO_TREE_FNAME
                        Taxonomy tree json file name.
  --imgs_dir IMGS_DIR   Training shape images path location.
  --epochs EPOCHS       Number of epochs to train for.
  --eval_fname EVAL_FNAME
                        [Optional] if provided, will run evaluation, this arg is the evaluation file name (see example).
  --output OUTPUT       Model output filename.
```


#### Cite this work
_TBD_
