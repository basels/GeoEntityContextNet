from collections import Counter
from datetime import timedelta, datetime
import json
import logging
from os.path import exists, join, getsize
import pandas as pd
import numpy as np
import pickle
from requests import get as get_request
import socket
from sys import stdout
from time import sleep, time
from tqdm import tqdm
from treelib import Node, Tree

hostname = socket.gethostname()

# ----------- logging -----------

logging.basicConfig(filename=f'logs/log_{hostname}__{datetime.now().strftime("%Y_%m_%d.%H_%M_%S")}.log',
                    filemode='w',
                    level=logging.INFO, # .DEBUG
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

log = logging.getLogger(__name__)

# ----------- general -----------


def save_dict_to_file(dict_, file_):
    with open(file_, "w") as file:
        json.dump(dict_, file, indent=2)


def load_dict_from_file(file_):
    with open(file_, "r") as file:
        dict_ = json.load(file)
    return dict_


def save_model(model, filepath):
    with open(filepath, "wb") as file:
        pickle.dump(model, file)

        
def load_model(filepath):
    with open(filepath, "rb") as file:
        model = pickle.load(file)
    return model

# ----------- tree ----------

def tree_to_dict(tree, node_id):
    node = tree.get_node(node_id)
    children = tree.children(node_id)
    node_dict = {"name": node.tag, "children": []}
    for child in children:
        node_dict["children"].append(tree_to_dict(tree, child.identifier))
    return node_dict

def dict_to_tree(tree, node_dict, parent_id=None):
    node_id = node_dict["name"]
    tree.create_node(node_dict["name"], node_id, parent=parent_id)
    for child_dict in node_dict["children"]:
        dict_to_tree(tree, child_dict, node_id)

# ----------- taxonomy matrix ----------

def get_common_ancestor(tree, node1_id, node2_id):
    # Get the paths from the nodes to the root
    path1 = tree.rsearch(node1_id)
    path2 = tree.rsearch(node2_id)
    # The paths are returned from the node to the root, so reverse them to traverse from the root
    path1 = reversed(list(path1))
    path2 = reversed(list(path2))
    
    last_common_node = None
    for n1, n2 in zip(path1, path2):
        if n1 == n2:
            last_common_node = n1
        else:
            break
    return last_common_node

def generate_taxonomy_matrix(tree, leaves):
    
    num_leaves = len(leaves)
    taxonomy_matrix = np.zeros((num_leaves, num_leaves))
    
    for i in range(num_leaves):
        for j in range(num_leaves):
            if i == j:
                taxonomy_matrix[i, j] = 0
            else:
                common_ancestor = get_common_ancestor(tree, leaves[i], leaves[j])
                taxonomy_matrix[i, j] = tree.depth() - tree.depth(node=common_ancestor)

    min_val = taxonomy_matrix.min()
    max_val = taxonomy_matrix.max()
    normalized_taxonomy_matrix = (taxonomy_matrix - min_val) / (max_val - min_val)
    
    return normalized_taxonomy_matrix