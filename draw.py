import altair as alt
import nx_altair as nxa
import pandas as pd
import numpy as np

from datetime import timedelta

from itertools import chain, product
from vega_datasets import data

import streamlit as st

import utils

PERSON = (
    "M1.7 -1.7h-0.8c0.3 -0.2 0.6 -0.5 0.6 -0.9c0 -0.6 "
    "-0.4 -1 -1 -1c-0.6 0 -1 0.4 -1 1c0 0.4 0.2 0.7 0.6 "
    "0.9h-0.8c-0.4 0 -0.7 0.3 -0.7 0.6v1.9c0 0.3 0.3 0.6 "
    "0.6 0.6h0.2c0 0 0 0.1 0 0.1v1.9c0 0.3 0.2 0.6 0.3 "
    "0.6h1.3c0.2 0 0.3 -0.3 0.3 -0.6v-1.8c0 0 0 -0.1 0 "
    "-0.1h0.2c0.3 0 0.6 -0.3 0.6 -0.6v-2c0.2 -0.3 -0.1 "
    "-0.6 -0.4 -0.6z"
)


def time_series(df, col):
    ''' make bar plot of time series data
        :param df:  Pandas DataFrame including "days" column
        :param col: column of DataFrame representing data to encode
        :return:    altair bar plot '''
    return alt.Chart(df).mark_bar().encode(
        x=alt.X('days', axis=alt.Axis(grid=False)),
        y=alt.Y(col, axis=alt.Axis(grid=False)),
        color=alt.value('#9467bd')
    ).properties(
        width=650,
        height=300
    )


def graph(graph, pos):
    ''' draw graph representation meta-cluster
        :param graph:   networkx graph to draw
        :param pos:     networkx layout of graph
        :return:        altair chart displaying graph '''
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


def templates(directory):
    ''' draw annotated text
        :param directory:   directory to look for InfoShield templates in
        :return:            altair annotated text '''
    to_write = utils.get_all_template_text(directory)

    utils.annotated_text(*to_write,
        scrolling=True,
        height=400
    )


def map(df):
    ''' generate map with ad location data
        :param df:  Pandas DataFrame with latitude, longitude, and count data
        :return:    altair map with ad counts displayed '''
    center, scale = utils.get_center_scale(df.lat, df.lon)

    countries = alt.topo_feature(data.world_110m.url, 'countries')
    base = alt.Chart(countries).mark_geoshape(
        fill='white',
        stroke='#DDDDDD'
    ).properties(
        width=900,
        height=400
    )

    agg_df = utils.aggregate_locations(df)

    scatter = alt.Chart(agg_df).mark_circle(
        color='#ff7f0e',
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


def bubble_chart(df, y, facet, tooltip):
    ''' create bubble chart 
        :param df:      Pandas DataFrame to display
        :param y:       column of DataFrame to use for bubble size
        :param facet:   column of DataFrame to create facet with
        :param tooltip: list of DataFrame columns to include in tooltip
        :return:        altair bubble chart '''
    return alt.Chart(df).mark_circle().encode(
        x=alt.X('days', axis=alt.Axis(grid=True)),
        y=alt.Y(y, axis=alt.Axis(grid=False, labels=False), title=None),
        color=alt.value('#17becf'),
        row=alt.Row(facet, title=None, header=alt.Header(labelAngle=-45)),
        tooltip=tooltip,
        size=alt.Size(y, scale=alt.Scale(range=[100, 500]))
    ).properties(
        width=450,
        height=400 / len(df)
    ).configure_facet(
        spacing=5
    ).configure_view(
        stroke=None
    )


def strip_plot(df, y, facet, tooltip):
    ''' create strip plot with heatmap
        :param df:      Pandas DataFrame to display
        :param y:       column of DataFrame to use for bubble size
        :param facet:   column of DataFrame to create facet with
        :param tooltip: list of DataFrame columns to include in tooltip
        :return:        altair strip plot '''

    delta = timedelta(days=1)
    date_range = np.arange(min(df.days) - delta, max(df.days) + 2*delta, delta)
    facet_s = facet.split(':')[0]

    label = df[facet_s].values[0]
    mini_df = pd.DataFrame([{'days': d, y: 0, facet_s: label} for d in date_range if d not in df.days.unique()])

    big_df = pd.concat([df, mini_df])

    return alt.Chart(big_df).mark_tick(binSpacing=0, thickness=6).transform_impute(
        impute=y,
        key='days',
        value=0,
        keyvals = date_range,
        groupby=[facet_s]
    ).encode(
        x=alt.X('days:T', axis=alt.Axis(grid=False), scale=alt.Scale(domain=[min(date_range), max(date_range)])),
        y=alt.Y(facet, axis=alt.Axis(grid=False, labels=True), title=None),
        color=alt.Color(y, scale=alt.Scale(scheme='purplered', type='sqrt')),
        tooltip=tooltip,
    ).properties(
        width=650,
        height=400
    ).configure_view(
        stroke=None
    ).configure_axis(
        labelFontSize=14,
        titleFontSize=14
    ).configure_legend(
        gradientLength=325,
        labelFontSize=14
    ).configure_axisX(
        labelAngle=-15
    )


def labeling_buttons(title):
    colors = ('#33cc33', '#ace600', '#e6e600', '#ff9900', '#ff3300')
    data = pd.DataFrame([{'id': i, 'color': c} for i, c in enumerate(colors)])
    brush = alt.selection_single(nearest=True, empty='none', fields=['id'])

    return alt.Chart(data).mark_point(
        filled=True,
        size=100
    ).encode(
        x=alt.X("id:O", axis=None),
        shape=alt.ShapeValue(PERSON),
        color=alt.condition(
            alt.datum.id <= brush.id,
            alt.Color('color:N', scale=None),
            alt.value('skyblue'))
    ).properties(
        width=400,
        height=100,
        title=title
    ).configure_view(
        strokeWidth=0
    ).add_selection(
        brush
    ).configure_title(
        fontSize=16
    )