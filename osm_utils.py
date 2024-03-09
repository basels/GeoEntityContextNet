from osm_taxonomy import *

from collections import Counter
from datetime import timedelta, datetime
from shapely import wkt
from shapely.validation import make_valid
import json
import logging
import lxml.etree
import matplotlib.pyplot as plt
from os.path import exists, join, getsize
import pandas as pd
import numpy as np
import pickle
from requests import get as get_request
import socket
import geopandas
from sys import stdout
from time import sleep, time
from tqdm import tqdm
from treelib import Node, Tree
from concurrent.futures import ProcessPoolExecutor
from sklearn import preprocessing

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

def save_model(model, filepath):
    with open(filepath, "wb") as file:
        pickle.dump(model, file)

        
def load_model(filepath):
    with open(filepath, "rb") as file:
        model = pickle.load(file)
    return model


def load_data(train_fname, taxo_tree_fname):
    train_df = pd.read_parquet(train_fname)
    with open(taxo_tree_fname, "r") as f:
        tree_dict = json.load(f)
    return train_df, tree_dict


def save_geodataframe(dataframe, filepath):
    dataframe['shp_centroid'] = dataframe['shp_centroid'].astype(str)
    dataframe.to_parquet(filepath)


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


# ----------- labels -----------


def get_curated_tag(candidate_tag, candidate_tags_ihi_others, valid_tags, taxonomy_tree=None):
    if candidate_tag in valid_tags:
        return candidate_tag
    else:
        # tag not in valid_tags, perhaps name should be more specific
        candidate_tag_parent = '__'.join(reversed(get_path_of_labels(candidate_tag, candidate_tags_ihi_others).split('--')[-2:]))
        if candidate_tag_parent in valid_tags:
            return candidate_tag_parent
        elif taxonomy_tree:
            # not found in list, let's look at ancestral nodes in taxonomy tree
            candidate_tag_paths_in_tree = None
            if taxonomy_tree.tree.contains(candidate_tag):
                candidate_tag_paths_in_tree = taxonomy_tree.get_tags_on_path_to_tags(candidate_tag)
            elif taxonomy_tree.tree.contains(candidate_tag_parent):
                candidate_tag_paths_in_tree = taxonomy_tree.get_tags_on_path_to_tags(candidate_tag_parent)
            
            if candidate_tag_paths_in_tree:
                shared_tags = list(set(candidate_tag_paths_in_tree).intersection(set(valid_tags)))
                if len(shared_tags) > 0:
                    return shared_tags[0]
    return None


# ----------- WKT -----------


def get_wkt_of_osm_relation_api(osm_relation_uri):
    '''
    example: for input 'https://www.openstreetmap.org/relation/3087669'
             we'll get the output: 'MULTIPOLYGON(((-117.1632189 33.8615651, ... )))'
    '''
    if '/relation/' in osm_relation_uri:
        relation_id = osm_relation_uri.split('/relation/')[-1]
        # seems like this loads it to their cache
        sleep(0.15)
        query = f'http://polygons.openstreetmap.fr/index.py?id={relation_id}'
        try:
            resp1 = get_request(query)
            if resp1.status_code == 200:
                sleep(0.15)
                query = f'http://polygons.openstreetmap.fr/get_wkt.py?id={relation_id}&params=0'
                resp2 = get_request(query)
                if resp2.status_code == 200:
                    # SRID=4326;MULTIPOLYGON(((-117.1632189 33.8615651, ... )))\n
                    return resp2.text.split(';')[-1].split('\n')[0] if ';' in resp2.text else None
        except:
            log.info(f'an exception occurred for relation/{relation_id}...')
    return None


def get_wkt_of_osm_way_api(osm_way_uri):
    '''
    example: for input 'https://www.openstreetmap.org/way/3087669'
             we'll get the output: 'LINESTRING(-118.3358412 34.1900233,-118.3358278 34.1906605)'
    '''
    if '/way/' in osm_way_uri:
        way_id = osm_way_uri.split('/way/')[-1]
        sleep(0.15)
        query = f'http://overpass-api.de/api/interpreter?data=[out:json];(way({way_id});%3E;);out;'
        resp = get_request(query)
        if resp.status_code == 200:     
            try:
                osm_data = resp.json()
                if 'elements' in osm_data:
                    # first build a dict of nodes, then use the order in the 'way' element
                    nodes_wkt = {}
                    is_poly = False
                    for idx, element in enumerate(osm_data['elements']):
                        if element['type'] == 'node':
                            nodes_wkt[element['id']] = str(element['lon']) + ' ' + str(element['lat'])
                        elif element['type'] == 'way':
                            if len(element['nodes']) >= 3 and element['nodes'][0] == element['nodes'][-1]:
                                wkt_string = 'POLYGON(('
                                is_poly = True
                            else:
                                wkt_string = 'LINESTRING('
                            
                            for node in element['nodes']:
                                wkt_string += nodes_wkt[node] + ','
                                
                            return wkt_string[:-1]+'))' if is_poly else wkt_string[:-1]+')'
            except ValueError as e:
                log.info("ValueError | " + str(e))
        return None


def get_wkt_string(x):
    if 'rel' in x.osm_id:
        return get_wkt_of_osm_relation_api('https://www.openstreetmap.org/' + x.osm_id)
    elif 'way' in x.osm_id:
        return get_wkt_of_osm_way_api('https://www.openstreetmap.org/' + x.osm_id)
    else:
        return f'POINT({x.lon} {x.lat})'


def get_wkt_image_name(osm_id, img_dir='data/train/img/'):
    img_name = f"{osm_id.replace('/', '_')}.jpg"
    full_path = join(img_dir, img_name)
    
    if not exists(full_path):
        return ''
    else:
        return img_name
    

def generate_image_from_osm_obj(args):
    osm_id, wkt_obj, img_dir = args
    if wkt_obj:
        fname_new = join(img_dir, f"{osm_id.replace('/', '_')}.jpg")
        if not exists(fname_new):
            wkt_df = pd.DataFrame({'g': [wkt_obj]})
            tmp_gdf = geopandas.GeoDataFrame(wkt_df, geometry='g')
            
            fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
            tmp_gdf.plot(ax=ax, color='black', linewidth=5)
            
            x_min, y_min, x_max, y_max = tmp_gdf.total_bounds
            
            # Validate the bounds
            bounds = [x_min, y_min, x_max, y_max]
            if not all(np.isfinite(b) for b in bounds):
                #print(f"Invalid bounds for OSM ID: {osm_id}, skipping...")
                plt.close(fig)
                return

            # Making the plot square
            max_extent = max(x_max - x_min, y_max - y_min) / 2
            centroid = (x_min + x_max)/2, (y_min + y_max)/2

            # Check for NaN or Inf in max_extent or centroid
            if not (np.isfinite(max_extent) and np.isfinite(centroid).all()):
                #print(f"Invalid max_extent or centroid for OSM ID: {osm_id}, skipping...")
                plt.close(fig)
                return

            ax.set_xlim([centroid[0] - max_extent, centroid[0] + max_extent])
            ax.set_ylim([centroid[1] - max_extent, centroid[1] + max_extent])
            
            ax.set_axis_off()
            ax.set_aspect('equal', adjustable='box')
            
            plt.tight_layout()
            fig.savefig(fname_new, dpi=100)
            plt.close(fig)
    return


def parallel_image_generation(g_ca, img_dir, n_jobs=-1):
    args = [(name, shp_wkt, img_dir) for name, shp_wkt in zip(g_ca['osm_id'], g_ca['shp_wkt'])] # g_ca.index
    
    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(tqdm(executor.map(generate_image_from_osm_obj, args), total=len(args)))


# ----------- Context Embedder -----------


class OSMNeighborhoodEmbedder():
    DIST_DELTA = 0.003 # 0.00075
    
    def __init__(self, geo_dataframe, valid_targets=[], osm_taxonomy_agent=None):
        
        self.valid_targets = list(set(valid_targets))
        
        self.gdf = geo_dataframe
        self.gdf_spatial_index = geo_dataframe.sindex
        
        self.nei_enc = preprocessing.LabelEncoder()
        self.nei_enc.fit(self.valid_targets)
        
        self.test_odf = None
        self.osm_taxo_agent = osm_taxonomy_agent

        
    def bitset_encode(self, dirty_neighborhood_labels, with_taxo_hier=False):
        
        # we can encode the ancestors on the taxonomy tree (if provided)
        if with_taxo_hier and self.osm_taxo_agent:
            all_path_labels = []
            for plabel in dirty_neighborhood_labels:
                plabel_paths_in_tree = self.osm_taxo_agent.get_tags_on_path_to_tags(plabel)
                all_path_labels += plabel_paths_in_tree
            dirty_neighborhood_labels = all_path_labels
        # only valid targets will be encoded
        input_neighborhood_labels = [k for k in dirty_neighborhood_labels if k in self.valid_targets]
        vector_of_label_ids = self.nei_enc.transform(input_neighborhood_labels)        
        C = len(self.nei_enc.classes_)
        fin_vec = np.zeros((1, C), dtype=int)
        if len(vector_of_label_ids) > 0:
            res_mat = np.eye(C)[vector_of_label_ids].astype(int)
            for onehot in res_mat:
                fin_vec += onehot
        return fin_vec[0]
    
    
    def encode_with_taxonomy_and_distance(self, neigh_labels_distances):
    
        inflated_neigh_labels_distances = []
        for plbl, pdist in neigh_labels_distances:
            plabel_paths_in_tree = self.osm_taxo_agent.get_tags_on_path_to_tags(plbl)
            inflated_neigh_labels_distances += [(k, pdist) for k in plabel_paths_in_tree]

        unique_lbls = set([k for k,_ in inflated_neigh_labels_distances])
        nl_df = pd.DataFrame(inflated_neigh_labels_distances, columns=['nei_l', 'nei_d'])

        fin_nei_dists = []
        for itm in unique_lbls:
            fin_nei_dists.append((itm, nl_df[nl_df['nei_l'] == itm]['nei_d'].min()))

        C = len(self.nei_enc.classes_)
        fin_vec = np.zeros((1, C), dtype=float)
        for k,v in fin_nei_dists:
            fin_vec += np.eye(C)[self.nei_enc.transform([k])].astype(float) * (self.DIST_DELTA - v)

        fin_vec[fin_vec == 0] = -1 * self.DIST_DELTA
        fin_vec /= self.DIST_DELTA
        
        return fin_vec[0]
    
    
    def get_osm_neighborhood(self, osm_centroid, store_geo=False):

        buffer = osm_centroid.buffer(self.DIST_DELTA, cap_style = 1)
        possible_matches_index = list(self.gdf_spatial_index.intersection(buffer.bounds))
        possible_matches = self.gdf.iloc[possible_matches_index]
        bbox_odf = possible_matches[possible_matches.intersects(buffer)]
        
        if len(bbox_odf) == 0:
            return list()
        
        bbox_odf['distance'] = bbox_odf.apply(lambda x: x.shp_wkt.distance(osm_centroid), axis=1)
        bbox_odf['color'] = bbox_odf.apply(lambda x: 'r' if x.shp_centroid == osm_centroid else '#FF9800', axis=1)
        
        if store_geo:
            self.test_odf = bbox_odf

        bbox_odf = bbox_odf[bbox_odf['color'] != 'r']
        if len(bbox_odf) == 0:
            return list()
        
        labels_and_distances = list(bbox_odf.apply(lambda x: (x.curated_label, x.distance), axis=1))
        return labels_and_distances
    
        
    def get_neighborhood_embedding(self, osm_centroid, store_geo=False, with_taxo_hier=False, with_distance=False):
        # get neighborhood instances
        osm_neighs__lbl_dist = self.get_osm_neighborhood(osm_centroid, store_geo)
        
        if with_distance:
            return self.encode_with_taxonomy_and_distance(osm_neighs__lbl_dist)
        else:
            osm_neigh_labels = [k for k,_ in osm_neighs__lbl_dist]
            return self.bitset_encode(osm_neigh_labels, with_taxo_hier=with_taxo_hier)
    
    def plot_most_recent_neighborhood(self, display_valid_only=False, display_annotation=False, ctx_zoom=15):
        
        if self.test_odf.empty:
            return
        
        sub_gdf = geopandas.GeoDataFrame(self.test_odf, geometry='shp_wkt', crs='epsg:4326')
        
        if display_valid_only:
            sub_gdf = sub_gdf[sub_gdf['curated_label'].isin(self.valid_targets)]
            
        sub_gdf = sub_gdf.to_crs(epsg=3857)
        
        #sub_gdf['coords'] = sub_gdf['shp_centroid'].apply(lambda x: x.coords[:])
        sub_gdf['coords'] = sub_gdf['shp_wkt'].apply(lambda x: x.centroid.coords[:])
        sub_gdf['coords'] = [coords[0] for coords in sub_gdf['coords']]
        
        ax = sub_gdf.plot(figsize=(16, 16), alpha=0.7, color=sub_gdf['color']) #, markersize=50
        if display_annotation:
            sub_gdf.apply(lambda x: ax.annotate(text=x.curated_label, xy=x.coords,
                                            horizontalalignment='center'), axis=1) # , fontsize=18
        
        ctx.add_basemap(ax, zoom=ctx_zoom)