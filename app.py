import os, sys
import streamlit as st

import draw, utils
import SessionState


def gen_page_content(state, graph, df, meta_clusters):
    title_cols = st.beta_columns(5)
    with title_cols[-1]:
        if st.button('View next cluster'):
            try:
                cluster = next(meta_clusters)
                state.cluster = cluster
                state.index += 1
            except StopIteration:
                state.is_stop = True
    with title_cols[0]:
        st.title('Suspicious Cluster #{}'.format(state.index+1))


    # on first iteration, before button press
    if state.is_first:
        cluster = next(meta_clusters) # TODO: change once scalable
        cluster = next(meta_clusters)
        state.index += 1
        state.cluster = cluster
        state.is_first = False

    # if we've processed all clusters, we show a static end page
    if state.is_stop:
        st.header("You've finished all examples from this dataset. Thank you!")
        st.balloons()
        return

    subdf = utils.get_subdf(df, state)
    cluster_features, features, metadata_features = utils.cluster_feature_extract(subdf)


    left_col, right_col = st.beta_columns((1.25, 3))

    with left_col:
        st.write(utils.BUTTON_STYLE, unsafe_allow_html=True)
        radio_val = st.radio('Which view would you like to see?', ['By cluster', 'By metadata'])

        top_n_params, bubble_params = utils.BY_CLUSTER_PARAMS if radio_val == 'By cluster' \
            else utils.BY_METADATA_PARAMS
        bubble_df = cluster_features if radio_val == 'By cluster' else metadata_features

        top_df = utils.top_n(bubble_df, **top_n_params)
        st.write(draw.bubble_chart(top_df, **bubble_params))

    with right_col:
        # template generation
        label = subdf['LSH label'].value_counts().idxmax()
        start_path = '../InfoShield/results/{}'.format(label)
        if not os.path.exists(start_path):
            start_path = './data/example.pkl'
        draw.templates(start_path)

    # hacky way to get padding between columns
    left_col, _, mid_col, _, right_col = st.beta_columns((1, 0.1, 1.5, 0.1, 2))

    with left_col:
        # meta-cluster stats table
        st.subheader('Meta-Cluster Stats')
        st.table(utils.basic_stats(graph, df, state.cluster, columns))

    with mid_col:
        # display features over time, aggregated forall clusters
        select_feature = st.selectbox('Choose a feature to look at over time', [f for f in features if f != 'days'])
        if select_feature:
            st.write(draw.time_series(features, select_feature))

    with right_col:
        # show map of ad locations
        st.write(draw.map(subdf))

    # Number input boxes take up the whole column space -- this makes them shorter
    st.subheader('Labeling: How likely is this to be...')
    new_cols = st.beta_columns(5)

    for col, cluster_type in zip(new_cols, ('Trafficking', 'Spam', 'Scam', 'Drug dealer', 'Other')):
        with col:
            st.number_input(cluster_type, 0.00)

    
# Generate content for app
st.set_page_config(layout='wide', page_title='Meta-Clustering Classification')
state_params = {'is_first': True,
    'index': 0,
    'cluster': set(),
    'is_stop': False
}
state = SessionState.get(**state_params)

with st.spinner('Processing data...'):
    filename = '../tiny-RANDOM.csv'
    columns = ['phone', 'email', 'social', 'image_id']
    df = utils.read_csv(filename)


    filename_stub = os.path.basename(filename).split('.')[0]

    graph = utils.construct_metaclusters(filename_stub, df, columns)
    meta_clusters = utils.gen_ccs(graph)

gen_page_content(state, graph, df, meta_clusters)