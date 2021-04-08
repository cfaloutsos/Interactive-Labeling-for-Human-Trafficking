import functools
import networkx as nx
import numpy as np
import os, sys
import pandas as pd
import pickle as pkl
import streamlit as st
import types

from annotated_text import annotated_text


### Params

BY_CLUSTER_PARAMS = ({
    'groupby': '# clusters per day',
    'sortby':  '# ads per day' 
}, {
    'y': '# ads per day',
    'facet': '# clusters per day:N',
    'tooltip': ['days', '# ads per day'],
})

BY_METADATA_PARAMS = ({
    'groupby': 'metadata',
    'sortby':  'count' 
}, {
    'y': 'count',
    'facet': 'metadata:N',
    'tooltip': ['days',  'count', 'metadata', 'type'],
})



BUTTON_STYLE = '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>'

### Generic utils
@st.cache#(show_spinner=False)
def read_csv(filename, keep_cols=[], rename_cols={}):
    df = pd.read_csv(filename)
    if keep_cols:
        df = df[keep_cols]

    if rename_cols:
        df = df.rename(columns=rename_cols)

    return df


@st.cache
def get_subdf(df, state, date_col='date_posted'):
    subdf = df[df['LSH label'].isin(state.cluster)].copy()
    subdf = gen_locations(subdf)

    subdf['location'] = [prettify_location(*tup) for tup in subdf[['city_id', 'country_id']].values]

    subdf[date_col] = pd.to_datetime(subdf[date_col], infer_datetime_format=True)
    subdf[date_col] = subdf[date_col].dt.date

    return subdf


@st.cache#(show_spinner=False)
def extract_field(field, cluster_label='LSH label'):
    field = field.dropna()
    if not len(field):
        return field

    return np.concatenate(field.apply(lambda val: str(val).split(';')).values)


@st.cache
def pretty_s(s):
    ''' return prettified version of string '''
    return '# {}s'.format(s.replace('_', ' '))


@st.cache#(show_spinner=False)
def basic_stats(graph, df, nodes, cols, cluster_label='LSH label'):
    subdf = df[df[cluster_label].isin(nodes)]

    metadata = {pretty_s(col): len(extract_field(subdf[col])) for col in cols}
    metadata['# ads'] = len(subdf)
    metadata['# clusters'] = len(subdf[cluster_label].unique())

    return pd.DataFrame(metadata, index=['Count']).T


@st.cache
def top_n(df, groupby, sortby, n=15):
    top_n = df.groupby(
        groupby
    ).sum(
        numeric_only=True
    ).sort_values(
        by=sortby,
        ascending=False
    ).index.values[:n]

    return df[df[groupby].isin(top_n)]


### Location data related functions
@st.cache
def get_center_scale(lat, lon):
    midpoint = lambda lst: (max(lst) + min(lst)) / 2

    scale = lambda lst, const: const*2 / (max(lst) - min(lst)) if max(lst) - min(lst) else 100

    center = midpoint(lon), midpoint(lat)

    scale_lat = scale(lat, 90)
    scale_lon = scale(lon, 180)

    return center, min(scale_lat, scale_lon) * 100


@st.cache
def gen_locations(df, count_only=True):
    # generate x&y coords for locations bar 
    cities_df = read_csv('~/grad_projects/data/aht_data/metadata/cities.csv',
        keep_cols=['id', 'xcoord', 'ycoord'],
        rename_cols={'xcoord': 'lat', 'ycoord': 'lon'})

    return pd.merge(df, cities_df, left_on='city_id', right_on='id', sort=False)


@st.cache
def prettify_location(city, country):
    cities_df = read_csv('~/grad_projects/data/aht_data/metadata/cities.csv')
    countries_df = read_csv('~/grad_projects/data/aht_data/metadata/countries.csv')

    # make pretty location string based on city, country       
    country_str = countries_df[countries_df.id == country].code.values[0]
    city_str = cities_df[cities_df.id == city].name.values[0]
    return ', '.join([city_str, country_str])


@st.cache
def aggregate_locations(df):
    agg_df = df.groupby(
        ['location'],
        as_index=False
    ).agg({
         'ad_id': 'count',
         'lat': 'mean',
         'lon': 'mean'
    })

    agg_df = agg_df.rename(columns = {'ad_id': 'count'})

    return agg_df


### Date related
@st.cache
def extract_field_dates(df, col_name, date_col):
    df = df.dropna()
    if not len(df):
        return pd.DataFrame(columns=['metadata', 'date_posted', 'count', 'type'])

    get_data = lambda row: [(val, row[date_col]) for val in str(row[col_name]).split(';')]
    concat_reduce = lambda data: functools.reduce(lambda x, y: x + y, data)

    # expand fields that have lists, so each is a row in df
    meta_df = pd.DataFrame(concat_reduce(df.apply(get_data, axis=1)), columns=df.columns)
    # aggregate by count
    meta_df = meta_df.groupby([col_name, date_col], as_index=False).size()
    # prettify df columns
    meta_df['type'] = col_name
    meta_df = meta_df.rename(columns={'size': 'count', col_name: 'metadata'})
    return meta_df

    #return pd.DataFrame(np.concatenate(df.apply(get_data, axis=1)), columns=df.columns)


@st.cache#(show_spinner=False)
def cluster_feature_extract(df, cluster_label='LSH label', date_col='date_posted', loc_col='city_id'):
    ''' extract important time-based features for a particular cluster '''
    def total(series):
        return len(extract_field(series))

    agg_dict = {name: total for name in ('ad_id', 'city_id', 'email', 'image_id')}
    rename_dict = {
        'ad_id': '# ads per day',
        'city_id': '# locations per day',
        'email': '# email accounts per day',
        'image_id': '# images per day',
        'LSH label': '# clusters per day',
        'social': '# social media tags per day',
        'date_posted': 'days'
    }

    by_cluster_df = df.groupby(
        [date_col, cluster_label],
        as_index=False,
        sort=False
    ).agg(agg_dict)

    agg_dict['LSH label'] = 'count'

    total_df = by_cluster_df.groupby(
        [date_col],
        as_index=False,
        sort=False
    ).agg(agg_dict)

    dfs = []
    for metadata in ('email', 'image_id', 'social', 'phone'):
        dfs.append(extract_field_dates(df[[metadata, date_col]], metadata, date_col))

    metadata_df = pd.concat(dfs).rename(columns={date_col: 'days'})

    return by_cluster_df.rename(columns=rename_dict), total_df.rename(columns=rename_dict), metadata_df


### Graph related utils
@st.cache#(show_spinner=False)
def construct_metaclusters(filename, df, cols, cluster_label='LSH label'):
    ''' construct metadata graph from dataframe already split into clusters
    @param df:              pandas dataframe containing ad info
    @param cols:            subset of @df.columns to link clusters by
    @param cluster_label:   column from @df.columns containing cluster label 
    @return                 nx graph, where each connected component is a meta-cluster '''

    pkl_filename = 'pkl_files/{}.pkl'.format(filename)
    if os.path.exists(pkl_filename):
        return pkl.load(open(pkl_filename, 'rb'))
    
    metadata_dict = defaultdict(list)
    metadata_graph = nx.Graph()
    

    for cluster_id, cluster_df in df.groupby(cluster_label):
        if cluster_id == -1:
            continue
        metadata_graph.add_node(cluster_id, num_ads=len(cluster_df))
        
        for name in cols:
            metadata_graph.nodes[cluster_id][name] = extract_field(cluster_df, name)
            
            for elem in metadata_graph.nodes[cluster_id][name]:
                edges = [(cluster_id, node) for node in metadata_dict[elem]]
                metadata_graph.add_edges_from(edges, type=name)
                metadata_dict[elem].append(cluster_id)
                    
    pkl.dump(metadata_graph, open(pkl_filename, 'wb'))
    return metadata_graph


@st.cache(hash_funcs={types.GeneratorType: id}, show_spinner=False)
def gen_ccs(graph):
    ''' return generator for connected components, sorted by size
        @param graph:   nx Graph 
        @return         generator of connected components '''

    components = sorted(nx.connected_components(graph), reverse=True, key=len)
    for component in components:
        print('# clusters', len(component))
        #if len(component) < 3:
        #    continue
        if len(component) > 50:
            pos = None
        else:
            pos = nx.kamada_kawai_layout(nx.subgraph(graph, component))
        yield component, pos


### Text annotation utils
@st.cache
def get_all_template_text(directory):
    if directory.endswith('.pkl'):
        pickled = pkl.load(open(directory, 'rb'))
        return get_template_text(*pickled, 0)
    to_write = []
    is_first = True
    for i, folder in enumerate(os.listdir(directory)):
        result_loc = '{}/{}/text.pkl'.format(directory, folder)
        if is_first:
            is_first = False
        else:
            to_write.append('<br><br>')

        pickled = pkl.load(open(result_loc, 'rb'))
        to_write += get_template_text(*pickled, i)

    return to_write

@st.cache
def get_template_text(template, ads, i):
    index_to_type = {
        -1: ('slot', '#faa'),
        0:  ('const', '#fea'),
        1:  ('sub', '#8ef'),
        2:  ('del', '#aaa'),
        3:  ('ins', '#afa'),
    }

    to_write = ['Template #{}: '.format(i), ' '.join(template)]

    for ad_index, ad in enumerate(ads):
        to_write.append('<br>Ad #{}'.format(ad_index+1))
        prev_type = None 
        for color_i, token in ad:
            curr_type, color = index_to_type[color_i]

            if curr_type == prev_type:
                prev_token = to_write[-1][0]
                to_write[-1] = ('{} {}'.format(prev_token, token), curr_type, color)
                continue

            prev_type = curr_type

            to_write.append((token, curr_type, color))

    return to_write
