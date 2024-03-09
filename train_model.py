from osm_utils import *
from osm_models import *

import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings

SEED = 42
log = logging.getLogger(__name__)
random.seed(SEED)
warnings.filterwarnings("ignore")
tqdm.pandas()

# ------- train script utils ---------

def setup_tree_and_encoder(train_df, tree_dict):
    taxo_embedding_tree = Tree()
    dict_to_tree(taxo_embedding_tree, tree_dict)

    train_df_red = train_df.sort_values('curated_label').drop_duplicates('curated_label')
    g_labels_encoder = preprocessing.LabelEncoder()
    g_labels_encoder.classes_ = train_df_red['curated_label'].values

    leaves = g_labels_encoder.classes_
    normalized_taxonomy_matrix = generate_taxonomy_matrix(taxo_embedding_tree, leaves)
    norm_taxonomy_mat = torch.FloatTensor(normalized_taxonomy_matrix)

    return norm_taxonomy_mat

# -------- main --------------

def main(args):
    # Load Data
    train_df, tree_dict = load_data(args.train_fname, args.taxo_tree_fname)
    norm_taxonomy_mat = setup_tree_and_encoder(train_df, tree_dict)
    log.info(f'Loaded training file: {args.train_fname}')
    log.info(f'Loaded Taxonomy Matrix:{args.taxo_tree_fname}\n{norm_taxonomy_mat}')
    
    # params
    num_epochs = int(args.epochs)
    opt_func = torch.optim.Adam
    batch_size = 32
    lr = 1e-5
    weight_decay = 0.05

    # dataset
    dataset = OSMShapeSpaceNeighborhoodDataset(dataframe=train_df,
                             img_dir=args.imgs_dir,
                             transform=transforms.Compose(OSMImageTransformations))
    log.info(f"Length of (Embedding) Train Data : {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Setup and Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OSMShapeSpatialNeighborhoodContrastive(num_of_spatial_feats=2, num_of_context_feats=train_df.iloc[0].neigh_emb.shape[0]).to(device)
    model.apply(weights_init_uniform_rule)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = NTXentLossWithTaxonomy(taxonomy_matrix=norm_taxonomy_mat.to(device), temperature=0.07)
    
    # multi GPUs if applicable
    device = torch.device("cuda:0")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # training loop
    history = []    
    for epoch in range(num_epochs):
        tmp_result = {'val_loss': 0, 'val_acc': 0}
    
        loss = training_contrastive_simple(model, dataloader, criterion, optimizer, device)
        tmp_result['train_loss'] = loss
        log.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
        history.append(tmp_result)

    log.info(f'Saving model: {args.output}')
    save_model(model, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Geo-Embedding model.")
    parser.add_argument("--train_fname", type=str, help="Training data file name.", required=True)
    parser.add_argument("--taxo_tree_fname", type=str, default='data/taxo_tree.json', help="Taxonomy tree json file name.")
    parser.add_argument("--imgs_dir", type=str, default='data/train/img/', help="Training shape images path location.")
    parser.add_argument("--epochs", type=str, default='10', help="Number of epochs to train for.")
    parser.add_argument("--output", type=str, default='geoemb_model.pkl', help="Model output filename.")

    args = parser.parse_args()
    main(args)
