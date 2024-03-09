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


# -------- main --------------

def main(args):
    
    # load osm dump
    blacklist = DEFAULT_BAD_OSM_TAGS
    log.info(f'Loading OSM (xml) file {args.input_osm}...')
    g_osm_data = get_osm_dictionary_from_xml(args.input_osm, bad_osm_tags=blacklist, get_geodata=True)

    g_osm_tagged_data = {k: v for (k, v) in g_osm_data.items() if 'tags' in v}
    del g_osm_data
    
    update_global_key_value_tag_counter(g_osm_tagged_data)
    g_new_osm_dict = build_new_structured_osm_dictionary(g_osm_tagged_data, blacklist)
    del g_osm_tagged_data

    g_osm_tagged_map = map(
        lambda x: (
            x[0],
            x[1]['tags'].get(MAIN_LABEL),
            x[1]['tags'].get(SEC_LABEL),
            x[1]['tags'].get(THI_LABEL),
            x[1].get('lon', None),
            x[1].get('lat', None)
        ),
        g_new_osm_dict.items()
    )
    df = pd.DataFrame(g_osm_tagged_map, columns=['osm_id', MAIN_LABEL, SEC_LABEL, THI_LABEL, 'lon', 'lat']) 
    del g_new_osm_dict, g_osm_tagged_map

    log.info(f"generating WKT representations")
    df['wkt'] = df.progress_apply(get_wkt_string, axis=1)
    
    df['shp_wkt']      = df['wkt'].progress_apply(wkt.loads)
    df['shp_wkt']      = df['shp_wkt'].progress_apply(lambda x: make_valid(x) if x else None)
    df['shp_type']     = df['shp_wkt'].progress_apply(lambda x: x.geom_type if x else None)
    df['shp_centroid'] = df['shp_wkt'].progress_apply(lambda x: x.centroid if x else None)
    df['shp_area']     = df['shp_wkt'].progress_apply(lambda x: x.area if x else None)
    df['shp_length']   = df['shp_wkt'].progress_apply(lambda x: x.length if x else None)
    
    parallel_image_generation(df, args.output_dir)
    log.info(f'Generated {len(df)} images for training at: {args.output_dir}')
    df['img'] = df.progress_apply(lambda x: get_wkt_image_name(x['osm_id'], args.output_dir), axis=1)

    log.info(f'Structuring a hierarchy of labels from taxonomy...')
    taxo_tree = get_taxo_tree(df, 10) # at least 10 instances for each label
    # taxo_tree.remove_invalid_tags(['unclassified', 'unofficial', 'multipolygon']) # example for additional filtering (post renaming)
    taxo_tree.build_taxonomy_tree()
    osm_targets = taxo_tree.get_all_tags_in_tree()

    log.info(f'Setting a single label for each instance / valid targets: {osm_targets}')
    df['curated_label'] = df.progress_apply(lambda x: get_curated_tag(x[MAIN_LABEL], x[SEC_LABEL], osm_targets, taxo_tree), axis=1)

    df = df[['osm_id', 'img', 'shp_type', 'shp_centroid', 'shp_area', 'shp_length', 'shp_wkt', 'curated_label']]
    df.dropna(subset=['shp_wkt', 'curated_label'], inplace=True)

    log.info(f"Creating the OSM Neighborhood Embedder")
    gdf = geopandas.GeoDataFrame(df, geometry='shp_wkt')
    on_embedder = OSMNeighborhoodEmbedder(geo_dataframe=gdf, valid_targets=osm_targets, osm_taxonomy_agent=taxo_tree)
    len(on_embedder.nei_enc.classes_)

    # only keep 'way' or 'relation' (for training)
    df = df[df['osm_id'].str.contains('way|rel')]

    labels_encoder = preprocessing.LabelEncoder()
    _ = labels_encoder.fit(df['curated_label'].unique())
    df['target_label_enc'] = labels_encoder.transform(df['curated_label'])

    df['neigh_emb'] = df.progress_apply(lambda x: on_embedder.get_neighborhood_embedding(x.shp_centroid, with_distance=True), axis=1)

    df = df.drop('shp_wkt', axis=1)
    df = df.set_index('osm_id')
    log.info(f'Saving file: {args.output_file}')
    save_geodataframe(df, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process the OSM data for the training of Geo-Embedding model.")
    parser.add_argument("--input_osm", type=str, help="OSM dump (xml) input filename.", required=True)
    parser.add_argument("--output_dir", type=str, default='data_old/test/', help="Output OSM images directory path.")
    parser.add_argument("--output_file", type=str, default='data_old/df_test.parquet', help="output filename (used for the training script).")

    args = parser.parse_args()
    main(args)
