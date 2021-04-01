import altair as alt
import networkx as nx
import pandas as pd
import streamlit as st
import types
import nx_altair as nxa

import time

from collections import defaultdict
from vega_datasets import data

import SessionState


@st.cache(show_spinner=False)
def read_csv(filename):
    return pd.read_csv(filename)

@st.cache(show_spinner=False)
def extract_field(df, field, cluster_label='LSH label'):
    return set([el for row in df[field] if type(row) != float for el in str(row).split(';')])

@st.cache(show_spinner=False)
def construct_metaclusters(df, cols, cluster_label='LSH label'):
    ''' construct metadata graph from dataframe already split into clusters
    @param df:              pandas dataframe containing ad info
    @param cols:            subset of @df.columns to link clusters by
    @param cluster_label:   column from @df.columns containing cluster label 
    @return                 nx graph, where each connected component is a meta-cluster '''
    
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
                    
    return metadata_graph


@st.cache(hash_funcs={types.GeneratorType: id}, show_spinner=False)
def gen_ccs(graph):
    ''' return generator for connected components, sorted by size
        @param graph:   nx Graph 
        @return         generator of connected components '''

    components = sorted(nx.connected_components(graph), reverse=True, key=len)
    for component in components:
        if len(component) < 3:
            continue
        yield component, nx.kamada_kawai_layout(nx.subgraph(graph, component))


def draw_time_feature(df, col):
    return alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('days', axis=alt.Axis(grid=False)),
        y=alt.Y(col, axis=alt.Axis(grid=False)),
    ).properties(
        width=600,
        height=300
    )


def draw_graph(graph, pos):
    return nxa.draw_networkx(
        graph, pos=pos,
        node_color='num_ads',
        cmap='tealblues',
        edge_color='grey'
        ).properties(
            width=500,
            height=500)


@st.cache
def get_center_scale(lat, lon):
    midpoint = lambda lst: (max(lst) + min(lst)) / 2

    scale = lambda lst, const: const*2 / (max(lst) - min(lst)) if max(lst) - min(lst) else 100

    center = midpoint(lon), midpoint(lat)

    #st.write('lat', max(lat), min(lat))
    #st.write('lon', max(lon), min(lon))

    scale_lat = scale(lat, 90)
    scale_lon = scale(lon, 180)

    return center, min(scale_lat, scale_lon) * 100


def draw_map(df, geo_col):
    get_geo = lambda index: [float(geo.split()[index]) for geo in df[geo_col].values]
    df['lat'] = get_geo(0)
    df['lon'] = get_geo(1)

    center, scale = get_center_scale(df.lat, df.lon)

    countries = alt.topo_feature(data.world_110m.url, 'countries')
    base = alt.Chart(countries).mark_geoshape(
        fill='white',
        stroke='#DDDDDD'
    ).properties(
        width=1200,
        height=600
    )

    scatter = alt.Chart(df).mark_circle(
        size=100,
        color='steelblue',
        fillOpacity=0.2
    ).encode(
        longitude='lon:Q',
        latitude='lat:Q',
        tooltip=['city', 'country']
    )

    return (base + scatter).project(
        'equirectangular',
        scale=scale,
        center=center
    )


def pretty_s(s):
    ''' return prettified version of string '''
    return '# {}s'.format(s.replace('_', ' '))

@st.cache(show_spinner=False)
def feature_extract(graph, df, nodes, cols, cluster_label='LSH label'):
    subdf = df[df[cluster_label].isin(nodes)]

    metadata = {pretty_s(col): len(extract_field(subdf, col)) for col in cols}
    metadata['# ads'] = len(subdf)
    metadata['# clusters'] = len(subdf[cluster_label].unique())

    return metadata


@st.cache(show_spinner=False)
def cluster_feature_extract(df, cluster, cluster_label='LSH label', date_col='crawl_date_ad', loc_col='geolocation'):
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
    st.title('Meta-Clustering Classification')

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

    st.header('Suspicious Cluster #{}'.format(state.index+1))
    features = cluster_feature_extract(df, state.cluster)

    left_col, right_col = st.beta_columns(2)

    with left_col:
        st.subheader('Meta-clustering graph')
        cluster = state.cluster
        subgraph = nx.subgraph(graph, state.cluster)
        st.write(draw_graph(subgraph, state.pos))

    with right_col:
        st.subheader('Basic Cluster Stats')
        st.write(feature_extract(graph, df, state.cluster, columns) )
        select_feature = st.selectbox('Choose a feature to look at over time', [f for f in features if f != 'days'])

        if select_feature:
            st.write(draw_time_feature(features, select_feature))

    subdf = df[df['LSH label'].isin(cluster)]
    st.write(draw_map(subdf, 'geolocation'))

    # Number input boxes take up the whole column space -- this makes them shorter
    new_cols = st.beta_columns(4)

    with new_cols[0]:
        st.subheader('Labeling: How likely is this to be...')
        for cluster_type in ('Trafficking', 'Spam', 'Scam', 'Drug dealer', 'Other'):
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
    columns = ['username', 'img_urls', 'phone_num']
    df = read_csv('../locanto_new_labels-normal_LSH_labels.csv')

    graph = construct_metaclusters(df, columns)
    meta_clusters = gen_ccs(graph)

gen_page_content(state, graph, df, meta_clusters)