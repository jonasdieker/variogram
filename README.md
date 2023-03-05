# Variogram Package

This was produced as part of my master's thesis which focussed on learning generative models from sparse data.

The package expects two dimensions in space and one in time. If 3D variogram computation is too costly,
one can choose to combine the two dimenions in space into a single spatial dimension.

## Installation

Install from source.

```console
git clone git@github.com:jonasdieker/variogram.git
cd variogram_analysis
pip install .
```

## Usage

### Load single csv file.

```python
import variogram

# specify file
dataset = variogram.Dataset("<file_name.csv>")

# loading single file
data = dataset.load_single_file_by_idx()
```

### Load entire folder of csv files.

```python
import variogram

# specify path to folder
dataset = variogram.Dataset("<path_to_folder>")

# load entire folder of files into DataFrame
data = dataset.load_all_files()
```

### Computing the variogram from loaded data.

```python
# compute the variogram, last coordinate axis needs to be time
v = variogram.Variogram(data, coordinate_axes=("x", "y", "time"), variables=("var1", "var2"))

# Either data is already detrended or use simple normal score detrending
v.detrend()

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
v_vis = VisualizeVariogram()
v_vis.read_varogram_from_file("<file_name.npy>")

# change variogram resolution to decrease noise in variogram
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


## Future Extensions
- [ ] Support visualizing when there are not exactly 2 variables.
- [ ] Support non-linear bins.
- [ ] More flexible support for dimensions/axis types.
- [ ] Faster performance by each thread having own copy of bins -> results in more RAM usage.