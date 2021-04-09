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
- [ ] Scalability (if we have > 3000 coarse clusters in a metacluster)

### Extra Features
- [ ] Create ad similarity metric
- [ ] Drill down into particular ads / templates
- [ ] Bubble chart -- on-click on a facet, populate its template
