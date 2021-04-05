# Interactive-Labeling-for-Human-Trafficking
An interactive visual system designed for domain experts to annotate suspicious clusters.

## Usage:
```
streamlit run app.py
```

## TODO:

### Basic app functionality
- [ ] Write labeling results when user moves on to another cluster
- [ ] Save state between runs -- only show to annotator if not previously clustered
- [ ] User input for csv, relevant columns
- [ ] Create publicly available toy example

### Extra Features
- [ ] On click of a node in metadata graph, display InfoShield template
- [ ] Prettify ooltips for meta-clustering graph
- [ ] More scalable graph drawing (awkward wait-time)
- [ ] Create ad similarity metric
- [ ] Drill down into particular ads
