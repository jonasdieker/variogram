from variogram.VisualizeVariogram import VisualizeVariogram
from variogram.Variogram import Variogram
from typing import List
import pandas as pd
import numpy as np
import xarray as xr
import itertools
from typing import Tuple, Dict


def load_config(yaml_file_config_path: str) -> Dict:
    config_path = os.path.join(yaml_file_config_path)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"\nLoaded config at: {config_path}.\n")
    return config


def setup_logger(log_root, now_string):
    logging.basicConfig(
        format="[%(levelname)s | %(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_root, f'{now_string}.log'))
        ])
    logger = logging.getLogger()
    return logger


def save_tuned_empirial_variogram_3d(vvis: VisualizeVariogram, view_range: List[int], file_path: str):
    """Convert hand-tuned variogram to dataframe and save."""

    lon_lag, lat_lag, time_lag = [], [], []
    u_semivariance, v_semivariance = [], []
    for lon in range(int(view_range[0]/vvis.variogram.res_tuple[0])):
        for lat in range(int(view_range[1]/vvis.variogram.res_tuple[1])):
            for time in range(int(view_range[2]/vvis.variogram.res_tuple[2])):
                u_semivariance.append(vvis.variogram.bins[lon, lat, time, 0])
                v_semivariance.append(vvis.variogram.bins[lon, lat, time, 1])
                lon_lag.append((lon+1)*vvis.variogram.lon_res)
                lat_lag.append((lat+1)*vvis.variogram.lat_res)
                time_lag.append((time+1)*vvis.variogram.t_res)

    df = pd.DataFrame({"lon_lag": lon_lag,
                       "lat_lag": lat_lag,
                       "t_lag": time_lag,
                       "u_semivariance": u_semivariance,
                       "v_semivariance": v_semivariance})
    df.to_csv(file_path, index=False)


def save_tuned_empirial_variogram_2d(vvis: VisualizeVariogram, view_range: List[int], file_path: str):
    """Convert hand-tuned variogram to dataframe and save."""

    space_lag, time_lag = [], []
    u_semivariance, v_semivariance = [], []
    for space in range(int(view_range[0]/vvis.variogram.res_tuple[0])):
        for time in range(int(view_range[1]/vvis.variogram.res_tuple[1])):
            u_semivariance.append(vvis.variogram.bins[space, time, 0])
            v_semivariance.append(vvis.variogram.bins[space, time, 1])
            space_lag.append((space+1)*vvis.variogram.space_res)
            time_lag.append((time+1)*vvis.variogram.t_res)

    df = pd.DataFrame({"space_lag": space_lag,
                       "t_lag": time_lag,
                       "u_semivariance": u_semivariance,
                       "v_semivariance": v_semivariance})
    df.to_csv(file_path, index=False)


def save_variogram_to_npy(variogram: Variogram, file_path: str):
    if variogram.bins is None:
        raise Exception("Need to build variogram first before you can save it!")

    data_to_save = {"bins": variogram.bins,
                    "bins_count": variogram.bins_count,
                    "res": variogram.res_tuple,
                    "units": variogram.units,
                    "detrend_metrics": variogram.data_statistics
                    }
    np.save(file_path, data_to_save)
    print(f"\nSaved variogram data to: {file_path}")


def sample_from_xr(data: xr.Dataset, num_samples, variables: Tuple[str, str] = ("u_error", "v_error")) -> pd.DataFrame:
    """Creates random sparse samples from dense data to be used in variogram computation.
    Note: Num of output points might differ from specified since NaN values are dropped.
    """
    lon_len = len(data["lon"].values)
    lat_len = len(data["lat"].values)
    time_len = len(data["time"].values)
    total_len = lon_len * lat_len * time_len

    idx = np.random.choice(list(range(total_len)), size=num_samples)

    lon = data["lon"].values.reshape(-1)
    lat = data["lat"].values.reshape(-1)
    time = data["time"].values.reshape(-1)

    axes = np.array(list(itertools.product(lon, lat, time)))[idx]
    lon = axes[:, 0]
    lat = axes[:, 1]
    time = axes[:, 2]
    u_error = data[variables[0]].values.reshape(-1)[idx]
    v_error = data[variables[1]].values.reshape(-1)[idx]

    data = pd.DataFrame({"lon": lon,
                         "lat": lat,
                         "time": time,
                         "u_error": u_error,
                         "v_error": v_error})

    data = data.dropna()
    return data
