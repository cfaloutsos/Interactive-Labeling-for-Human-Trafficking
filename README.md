# Interactive-Labeling-for-Human-Trafficking
An interactive visual system designed for domain experts to annotate suspicious clusters.

## Usage:
```
streamlit run app.py
```

## TODO:

### Basic app functionality
- [ ] Connect with InfoShield templates
- [ ] Write labeling results when user moves on to another cluster
- [ ] Save state between runs -- only show to annotator if not previously clustered
- [ ] User input for csv, relevant columns
- [ ] Create publicly available toy example
- [ ] Redesign layout to fit on one page (wide view)
- [ ] Fill in more M.O.s

### Extra Features
- [ ] On click of a node in metadata graph, display InfoShield template
- [ ] Tooltips for meta-clustering graph
- [ ] Proper scaling for the world map view + better colors
- [ ] More scalable graph drawing (awkward wait-time)
- [ ] Create ad similarity metric
- [ ] Drill down into particular ads
