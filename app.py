import altair as alt
import networkx as nx
import os, sys
import pandas as pd
import pickle as pkl
import streamlit as st
import types
import nx_altair as nxa

import time

from collections import defaultdict
from itertools import chain
from annotated_text import annotated_text
from vega_datasets import data

import SessionState


@st.cache(show_spinner=False)
def read_csv(filename):
    return pd.read_csv(filename)


@st.cache(show_spinner=False)
def extract_field(df, field, cluster_label='LSH label'):
    return set([el for row in df[field] if type(row) != float for el in str(row).split(';')])


@st.cache(show_spinner=False)
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


@st.cache
def gen_locations(df, count_only=True):
    # geberate x&y coords for locations bar 
    cities_df = read_csv('~/grad_projects/data/aht_data/metadata/cities.csv')
    countries_df = read_csv('~/grad_projects/data/aht_data/metadata/countries.csv')

    lat, lon = [], []
    for city in df.city_id.values:
        row = cities_df[cities_df.id == city].iloc[0]
        lat.append(row.xcoord)
        lon.append(row.ycoord)

    return lat, lon

@st.cache
def prettify_location(city, country):
    cities_df = read_csv('~/grad_projects/data/aht_data/metadata/cities.csv')
    countries_df = read_csv('~/grad_projects/data/aht_data/metadata/countries.csv')

    # make pretty location string based on city, country       
    country_str = countries_df[countries_df.id == country].code.values[0]
    city_str = cities_df[cities_df.id == city].name.values[0]
    return ', '.join([city_str, country_str])


@st.cache(hash_funcs={types.GeneratorType: id}, show_spinner=False)
def gen_ccs(graph):
    ''' return generator for connected components, sorted by size
        @param graph:   nx Graph 
        @return         generator of connected components '''

    components = sorted(nx.connected_components(graph), reverse=True, key=len)
    for component in components:
        print(len(component))
        if len(component) < 3:
            continue
        if len(component) > 50:
            pos = None
        else:
            pos = nx.kamada_kawai_layout(nx.subgraph(graph, component))
        yield component, pos


def draw_time_feature(df, col):
    return alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('days', axis=alt.Axis(grid=False)),
        y=alt.Y(col, axis=alt.Axis(grid=False)),
    ).properties(
        width=650,
        height=300
    )


@st.cache(hash_funcs={dict: lambda _: None})
def draw_graph(graph, pos):
    node_attributes = set(chain.from_iterable(d.keys() for *_, d in graph.nodes(data=True)))
    edge_attributes = set(chain.from_iterable(d.keys() for *_, d in graph.edges(data=True)))
    return nxa.draw_networkx(
        graph, pos=pos,
        node_color='num_ads',
        cmap='tealblues',
        edge_color='grey',
        node_tooltip=list(node_attributes),
        edge_tooltip=list(edge_attributes)
        ).properties(
            width=450,
            height=400
        ).configure_view(
            strokeWidth=0
        )


def draw_templates(directory):
    to_write = get_all_template_text(directory)

    annotated_text(*to_write,
        scrolling=True,
        height=400
    )

@st.cache
def get_all_template_text(directory):
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

        

@st.cache
def get_center_scale(lat, lon):
    midpoint = lambda lst: (max(lst) + min(lst)) / 2

    scale = lambda lst, const: const*2 / (max(lst) - min(lst)) if max(lst) - min(lst) else 100

    center = midpoint(lon), midpoint(lat)

    scale_lat = scale(lat, 90)
    scale_lon = scale(lon, 180)

    return center, min(scale_lat, scale_lon) * 100


def draw_map(df):
    #get_geo = lambda index: [float(geo.split()[index]) for geo in df[geo_col].values]
    lat, lon = gen_locations(df)

    df['location'] = [prettify_location(*tup) for tup in df[['city_id', 'country_id']].values]

    df['lat'] = lat
    df['lon'] = lon

    center, scale = get_center_scale(lat, lon)

    countries = alt.topo_feature(data.world_110m.url, 'countries')
    base = alt.Chart(countries).mark_geoshape(
        fill='white',
        stroke='#DDDDDD'
    ).properties(
        width=900,
        height=400
    )

    agg_df = df.groupby(['location'], as_index=False).agg({'ad_id': 'count', 'lat': 'mean', 'lon': 'mean'})
    agg_df=agg_df.rename(columns = {'ad_id': 'count'})

    scatter = alt.Chart(agg_df).mark_circle(
        color='#7D3C98',
        fillOpacity=.5,
    ).encode(
        size=alt.Size('count:Q', scale=alt.Scale(range=[100, 500])),
        longitude='lon:Q',
        latitude='lat:Q',
        tooltip=['location', 'count']
    )

    return (base + scatter).project(
        'equirectangular',
        scale=scale,
        center=center
    )

@st.cache
def pretty_s(s):
    ''' return prettified version of string '''
    return '# {}s'.format(s.replace('_', ' '))

@st.cache(show_spinner=False)
def basic_stats(graph, df, nodes, cols, cluster_label='LSH label', index=0):
    subdf = df[df[cluster_label].isin(nodes)]

    metadata = {pretty_s(col): len(extract_field(subdf, col)) for col in cols}
    metadata['# ads'] = len(subdf)
    metadata['# clusters'] = len(subdf[cluster_label].unique())

    index = ['Cluster #{}'.format(index)]
    return pd.DataFrame(metadata, index=index).T


@st.cache(show_spinner=False)
def cluster_feature_extract(df, cluster, cluster_label='LSH label', date_col='date_posted', loc_col='city_id'):
    ''' extract important time-based features for a particular cluster '''

    subdf = df[df[cluster_label].isin(cluster)]

    subdf[date_col] = pd.to_datetime(subdf[date_col], infer_datetime_format=True)
    days = pd.date_range(min(subdf[date_col]), max(subdf[date_col]))

    days_dfs = [subdf[subdf[date_col] == d] for d in days]

    features = {
        '# ads per day': [len(day_df) for day_df in days_dfs],
        '# locations per day': [len(day_df[loc_col].unique()) for day_df in days_dfs],
        'days': days
    }

    return pd.DataFrame(features)


def gen_page_content(state, graph, df, meta_clusters):
    st.title('Suspicious Cluster #{}'.format(state.index+1))

    # to show the first cluster before "View next cluster" button press
    if state.is_first:
        cluster, pos = next(meta_clusters)
        state.cluster = cluster
        state.pos = pos
        state.is_first = False

    # if we've processed all clusters, we show a static end page
    if state.is_stop:
        st.header("You've finished all examples from this dataset. Thank you!")
        st.balloons()
        return

    features = cluster_feature_extract(df, state.cluster)

    left_col, right_col = st.beta_columns((1, 3))

    with left_col:
        # graph data
        st.subheader('Meta-clustering graph')
        cluster = state.cluster
        subgraph = nx.subgraph(graph, state.cluster)
        if len(cluster) < 100:
            mc_graph = draw_graph(subgraph, state.pos)
            select = alt.selection_single()
            st.write(mc_graph.add_selection(select))


    with right_col:
        # template generation
        subdf = df[df['LSH label'].isin(cluster)]
        label = subdf['LSH label'].value_counts().idxmax()
        start_path = '../InfoShield/results/{}'.format(142.0)
        #start_path = '../InfoShield/results/{}'.format(int(label))
        draw_templates(start_path)



    left_col, _, mid_col, _, right_col = st.beta_columns((1, 0.1, 1.5, 0.1, 2))

    with left_col:
        # basic stats table
        st.subheader('Meta-Cluster Stats')
        st.table(basic_stats(graph, df, state.cluster, columns, index=state.index+1))

    with mid_col:
        # features over time
        select_feature = st.selectbox('Choose a feature to look at over time', [f for f in features if f != 'days'])
        if select_feature:
            st.write(draw_time_feature(features, select_feature))

    with right_col:
        # map
        st.write(draw_map(subdf))




    # Number input boxes take up the whole column space -- this makes them shorter
    st.subheader('Labeling: How likely is this to be...')
    new_cols = st.beta_columns(5)


    for col, cluster_type in zip(new_cols, ('Trafficking', 'Spam', 'Scam', 'Drug dealer', 'Other')):
        with col:
            st.number_input(cluster_type, 0.00)

    with new_cols[-1]:
        if st.button('View next cluster'):
            try:
                cluster, pos = next(meta_clusters)
                state.cluster = cluster
                state.pos = pos
                state.index += 1
            except StopIteration:
                state.is_stop = True

    
# Generate content for app
st.set_page_config(layout='wide', page_title='Meta-Clustering Classification')
state_params = {'is_first': True,
    'index': 0,
    'cluster': set(),
    'pos': None,
    'is_stop': False
}
state = SessionState.get(**state_params)

with st.spinner('Processing data...'):
    #filename = '../RANDOM-CONCATENATED-normal_LSH_labels.csv'
    filename = '../tiny-RANDOM.csv'
    columns = ['phone', 'email', 'social', 'image_id']
    #columns = ['username', 'phone_num']
    df = read_csv(filename)


    filename_stub = os.path.basename(filename).split('.')[0]

    graph = construct_metaclusters(filename_stub, df, columns)
    meta_clusters = gen_ccs(graph)

gen_page_content(state, graph, df, meta_clusters)