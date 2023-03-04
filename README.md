# Variogram Analysis Package

This was produced as part of my master's thesis which focussed on learning generative models from sparse data.

## Installation

Install from source

```console
    git clone <link>
    cd variogram_analysis
    pip install .
```

## Usage

```python
import variogram as v
import pandas as pd

# load prepared sparse data from .csv file
data = pd.read_csv("your_data.csv")

# compute the variogram
V = v.compute_varogram(data)

# visualize variogram slices in all dimensions
V_vis = v.VisualizeVariogram(V)
```
