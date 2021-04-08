import altair as alt
import nx_altair as nxa

from itertools import chain
from vega_datasets import data

import utils


def time_series(df, col):
    return alt.Chart(df).mark_bar().encode(
        x=alt.X('days', axis=alt.Axis(grid=False)),
        y=alt.Y(col, axis=alt.Axis(grid=False)),
    ).properties(
        width=650,
        height=300
    )


def graph(graph, pos):
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
    to_write = utils.get_all_template_text(directory)

    utils.annotated_text(*to_write,
        scrolling=True,
        height=400
    )


def map(df):
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


def bubble_chart(df, y, facet, tooltip):
    return alt.Chart(df).mark_circle().encode(
        x=alt.X('days', axis=alt.Axis(grid=True)),
        y=alt.Y(y, axis=alt.Axis(grid=False, labels=False), title=None),
        color=alt.Color(facet, legend=None),
        row=alt.Row(facet, title=None, header=alt.Header(labelAngle=-45)),
        tooltip=tooltip,#['days', 'count', 'metadata', 'type'],
        size=alt.Size(y, scale=alt.Scale(range=[100, 500]))
    ).properties(
        width=450,
        height=400 / len(df)
    ).configure_facet(
        spacing=5
    ).configure_view(
        stroke=None
    )
