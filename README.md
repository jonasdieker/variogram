# Variogram Package

This was produced as part of my master's thesis which focussed on learning generative models from sparse data.

## Installation

Install from source.

```console
git clone git@github.com:jonasdieker/variogram.git
cd variogram_analysis
pip install .
```

## Usage

### Computing the variogram from a csv file containing sparse data.

```python
import variogram

# make sure config.yaml file contains all meta-data, including path of .csv file
config = load_config("config.yaml")

# load data
dataset = variogram.Dataset(config["data_file"])

# compute the variogram
v = variogram.Variogram(data)

# Either data is already detrended or use simple normal score detrending
v.detrend(detrend_var="lat")

# compute variogram
v.build_variogram(res_tuple=(1, 1, 1), num_workers=2, chunk_size=1e6, detrend=False)

# save variogram to file
variogram.save_variogram_to_npy(v, file_path)
```

### Visualizing the variogram in all dimensions.

```python
# visualize variogram form existing variogram object
v_vis = VisualizeVariogram(v)

# or visualize variogram from file
v_vis = VisualizeVariogram.read_varogram_from_file(<file_name.npy>)

# change variogram resolution
v_vis.decrease_variogram_res(res_tuple=(5, 5, 5))

# show histogram of bins
v_vis.plot_histograms()

# show variogram sliced in each dimension
v_vis.plot_variograms()
```

## Further Notes

Computing the variogram is quite costly in both space and time.

If there are a lot of points the code can take a long time to run, even using multiple workers. In this case it is advised to run the computation in tmux.

If the code breaks (e.g. produces a seg fault) it is likely due to insufficient memory. In this case the following should be tried:

- Reduce the range of data for which the variogram is being compute.
- Reduce the resolution of the variogram computation (i.e. increase *res_tuple* values).
- Reduce the number of dimensions of the variogram.