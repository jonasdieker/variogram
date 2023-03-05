from variogram.IndexPairGenerator import IndexPairGenerator

import pandas as pd
import numpy as np
from typing import Tuple, List, AnyStr, Dict
import multiprocessing as mp
import ctypes as c
import logging


class Variogram:
    """
    Handles all the required variogram computation functionality to compute 3D variograms.
    """

    def __init__(self, data: pd.DataFrame=None, coordinate_axes: Tuple[str] = ("x", "y", "time"), variables: Tuple[str] = ("u_error", "v_error")):
        self.data = data
        if data is not None:
            self.data.dropna(inplace=True)
        self.coordinate_axes = coordinate_axes
        if len(self.coordinate_axes) != 3:
            raise ValueError("Need to specify 3 coordinate axes to construct a 3D variogram.")
        self.variables = variables
        self.bins = None
        self.bins_count = None
        self.data_statistics = None
        self.units = None

    def detrend(self) -> Dict[str, List[float]]:
        """Variogram analysis assumes (second-order) stationarity, therefore detrending is needed.
        
        This method can performs a simple detrending strategy by subtracting the mean and dividing
        by standard deviation."""

        data_statistics = {}
        for var in self.variables:
            stats = []
            stats.append(self.data[var].mean())
            stats.append(np.sqrt(self.data[var].var()))
            data_statistics |= {var: stats}

        # detrend data
        for var in data_statistics:
            self.data[var] = (self.data[var] - data_statistics[var][0])/data_statistics[var][1]

        self.data_statistics = data_statistics


    def build_variogram(self, res_tuple: Tuple[float], num_workers: int, chunk_size: int, cross_buoy_pairs_only: bool=False,
                        is_3d: bool=True, units: str="degrees", logger: logging=None) -> Tuple[np.ndarray, np.ndarray]:

        """Computes the variogram of self.data.
        
        It finds all possible pairs of points. Then computes the lag value in each axis
        and the variogram value for u and v errors.

        This method uses a generator if there are too many index pairs to hold in RAM.
        
        res_tuple - resolution in units of column and correspond directly to self.coordinate_axes
        num_workers - workers for multi-processing
        chunk_size - lower number if less memory
        cross_buoy_pairs_only - whether or not to use point pairs from the same buoy
        is_3d - take 3d variogram vs 2d variogram where space dims combined by average
        """

        self.units, self.is_3d = units, is_3d
        if is_3d:
            assert len(res_tuple) == 3, "Need res_tuple of length 3."
        else:
            assert len(res_tuple) == 2, "Need res_tuple of length 2."
        self._variogram_setup(res_tuple)

        # setup generator to generate index pairs
        n = self.data.shape[0]
        gen = IndexPairGenerator(n, int(chunk_size))

        # make bins
        self.bins = np.zeros((*self.bin_sizes, len(self.variables)), dtype=np.float32)
        self.bins_count = np.zeros_like(self.bins, dtype=np.int32)

        # convert relevant columns to numpy before accessing
        converted_coord_axes = []
        for axis in self.coordinate_axes:
            converted_coord_axes.append(self.data[axis].to_numpy(dtype=np.float32))

        # convert variables
        converted_variables = []
        for var in self.variables:
            converted_variables.append(self.data[var].to_numpy(dtype=np.float32))

        buoy_vector = None
        if cross_buoy_pairs_only:
            buoy_vector = self.data["buoy"].to_numpy()

        # setup shared arrays for workers
        shared_bins = to_shared_array(self.bins, c.c_float)
        self.bins = to_numpy_array(shared_bins, self.bins.shape)

        shared_bins_count = to_shared_array(self.bins_count, c.c_int32)
        self.bins_count = to_numpy_array(shared_bins_count, self.bins_count.shape)

        # setup mp stuff
        q = mp.Queue(maxsize=num_workers)
        iolock = mp.Lock()
        chunking_method_map = {True: self._calculate_chunk_3d, False: self._calculate_chunk_2d}
        pool = mp.Pool(num_workers,
                       initializer=chunking_method_map[is_3d],
                       initargs=(q, *converted_coord_axes, converted_variables, buoy_vector, iolock))

        # iterate over generator to get relevant indices
        number_of_pairs = int((n**2)/2 - n/2)
        running_sum = 0
        iteration = 0
        while True:
            indices = next(gen)
            if cross_buoy_pairs_only:
                indices = self.mask_pairs_from_same_buoy(indices, buoy_vector)
            if len(indices[0]) == 0:
                break
            q.put(indices)
            with iolock:
                running_sum += len(indices[0])
                if iteration % 10 == 0 and iteration != 0 and logger is not None:
                    logger.info(f"Iteration: {iteration}, Estimated {round(100*(running_sum/number_of_pairs),2)}% of pairs finished.")
                iteration += 1

        for _ in range(num_workers):
            q.put(None)

        pool.close()
        pool.join()
        
        # divide bins by the count + divide by 2 for semi-variance (special func to avoid dividing by zero)
        self.bins = np.divide(self.bins, self.bins_count, out=np.zeros_like(self.bins), where=self.bins_count != 0)/2
        return self.bins, self.bins_count

    def _calculate_chunk_3d(self, q: mp.Queue, lon: List[int], lat: List[int], time: List[int],\
        variables: List[List[float]], buoy_vector: List[str], iolock: mp.Lock) -> None:
        """Used in multiprocessing to compute the variogram values of pairs of points according to
        the indices."""

        while True:
            # get values for chunk
            indices = q.get()
            if indices is None:
                break
            idx_i = np.array(indices[0])
            idx_j = np.array(indices[1])

            # get lags in degrees and divide by bin resolution to get bin indices
            if self.units == "degrees":
                lon_lag = np.floor(np.absolute(lon[idx_i] - lon[idx_j])/self.lon_res).astype(int)
                lat_lag = np.floor(np.absolute(lat[idx_i] - lat[idx_j])/self.lat_res).astype(int)
            elif self.units == "km":
            # convert lags from degrees to kilometres
                pts1 = np.hstack((lon[idx_i].reshape(-1, 1), lat[idx_i].reshape(-1, 1)))
                pts2 = np.hstack((lon[idx_j].reshape(-1, 1), lat[idx_j].reshape(-1, 1)))
                lon_lag, lat_lag = convert_degree_to_km(pts1, pts2)
                # convert to bin indices
                lon_lag = np.floor(np.absolute(lon_lag)/self.lon_res).astype(int)
                lat_lag = np.floor(np.absolute(lat_lag)/self.lat_res).astype(int)

            t_lag = np.floor((np.absolute(time[idx_i] - time[idx_j])/self.t_res).astype(float)).astype(int)

            squared_diffs = []
            for var in variables:
                squared_diffs.append(np.square(var[idx_i] - var[idx_j]))
            squared_diff = np.moveaxis(np.array(squared_diffs), 0, -1)

            with iolock:
                # from: https://stackoverflow.com/questions/51092737/vectorized-assignment-in-numpy
                # add to relevant bin
                np.add.at(self.bins, (lon_lag, lat_lag, t_lag), squared_diff)
                # add to bin count
                np.add.at(self.bins_count, (lon_lag, lat_lag, t_lag), np.ones(len(variables)))

    def _calculate_chunk_2d(self, q: mp.Queue, lon: List[int], lat: List[int], time: List[int], variables: List[List[float]],
                            buoy_vector: List[str], iolock: mp.Lock) -> None:
        """Used in multiprocessing to compute the variogram values of pairs of points according to
        the indices."""

        while True:
            # get values for chunk
            indices = q.get()
            if indices is None:
                break
            idx_i = np.array(indices[0])
            idx_j = np.array(indices[1])

            # get lags in degrees and divide by bin resolution to get bin indices
            if self.units == "degrees":
                lon_lag = np.absolute(lon[idx_i] - lon[idx_j])
                lat_lag = np.absolute(lat[idx_i] - lat[idx_j])
                space_lag = np.floor(0.5*(lon_lag + lat_lag) / self.space_res).astype(int)
            elif self.units == "km":
                # convert lags from degrees to kilometres
                pts1 = np.hstack((lon[idx_i].reshape(-1, 1), lat[idx_i].reshape(-1, 1)))
                pts2 = np.hstack((lon[idx_j].reshape(-1, 1), lat[idx_j].reshape(-1, 1)))
                lon_lag, lat_lag = convert_degree_to_km(pts1, pts2)
                # take average and convert to bin indices
                space_lag = np.floor(0.5*(lon_lag + lat_lag) / self.space_res).astype(int)

            t_lag = np.floor((np.absolute(time[idx_i] - time[idx_j]) / self.t_res).astype(float)).astype(int)

            squared_diffs = []
            for var in variables:
                squared_diffs.append(np.square(var[idx_i] - var[idx_j]))
            squared_diff = np.moveaxis(np.array(squared_diffs), 0, -1)

            with iolock:
                # from: https://stackoverflow.com/questions/51092737/vectorized-assignment-in-numpy
                # add to relevant bin
                np.add.at(self.bins, (space_lag, t_lag), squared_diff)
                # add to bin count
                np.add.at(self.bins_count, (space_lag, t_lag), np.ones(len(variables)))

    def _variogram_setup(self, res_tuple: Tuple[float]):
        """Helper method to determine the number of bins."""

        x_coord, y_coord, time_coord = self.coordinate_axes

        # convert time axis to datetime
        self.data[time_coord] = pd.to_datetime(self.data[time_coord])
        # find earliest time/date and subtract from time column and convert to hours
        earliest_date = self.data[time_coord].min()
        self.data[time_coord] = self.data[time_coord].apply(
            lambda x: (x - earliest_date).seconds // 3600 + (x - earliest_date).days * 24)

        # calculate bin sizes from known extremal values and given resolutions
        self.t_bins = np.ceil(
            (self.data[time_coord].max() - self.data[time_coord].min() + 1) / res_tuple[-1]).astype(int)

        # find extremal values for space
        max_x, min_x = self.data[x_coord].max(), self.data[x_coord].min()
        max_y, min_y = self.data[y_coord].max(), self.data[y_coord].min()

        if self.units == "degrees":
            max_x_lag = max_x - min_x
            max_y_lag = max_y - min_y
        elif self.units == "km":
            max_y_lag = (110.574 * max_y) - (110.574 * min_y)
            max_x_lag = (111.320 * max_x) - (111.320 * min_x)
        else:
            raise NameError("Unknown units specified!")
    
        self.res_tuple = res_tuple
        if self.is_3d:
            self.x_res, self.y_res, self.t_res = res_tuple
            self.x_bins = np.ceil(abs(max_x_lag/self.x_res)+5).astype(int)
            self.y_bins = np.ceil(abs(max_y_lag/self.y_res)+5).astype(int)
            self.bin_sizes = (self.x_bins, self.y_bins, self.t_bins)
        else:
            self.space_res, self.t_res = res_tuple
            self.space_bins = np.ceil((0.5*(max_y_lag + max_x_lag) / self.space_res) + 1).astype(int)
            self.bin_sizes = (self.space_bins, self.t_bins)

    def mask_pairs_from_same_buoy(self, indices, buoy_vector):
        """build mask to eliminate pairs from same buoy.
        """
        # unpack indices into vectors
        idx_i = np.array(indices[0])
        idx_j = np.array(indices[1])

        # get buoy vectors
        buoy_vector_i = buoy_vector[idx_i]
        buoy_vector_j = buoy_vector[idx_j]
        # map each buoy name string to unique integer
        _, integer_mapped = np.unique([buoy_vector_i, buoy_vector_j], return_inverse=True)
        buoy_vector_mapped_i = integer_mapped[:round(len(integer_mapped) / 2)]
        buoy_vector_mapped_j = integer_mapped[round(len(integer_mapped) / 2):]
        # if integer in i and j is the same, then both points from same buoy
        residual = buoy_vector_mapped_i - buoy_vector_mapped_j
        mask_same_buoy = residual != 0
        mask_idx = np.where(mask_same_buoy == True)  # need == here!

        # only select pairs from different buoys
        idx_i = idx_i[mask_idx]
        idx_j = idx_j[mask_idx]
        return np.array([idx_i, idx_j])


#------------------------ Helper Funcs -----------------------#


def to_shared_array(arr, ctype):
    shared_array = mp.Array(ctype, arr.size, lock=False)
    temp = np.frombuffer(shared_array, dtype=arr.dtype)
    temp[:] = arr.flatten(order='C')
    return shared_array


def to_numpy_array(shared_array, shape):
    """Create a numpy array backed by a shared memory Array."""
    arr = np.ctypeslib.as_array(shared_array)
    return arr.reshape(shape)


def create_shared_array_from_np(arr, ctype):
    """Combines two functions, implemented due to repetition"""
    shared_array = to_shared_array(arr, ctype)
    output = to_numpy_array(shared_array, arr.shape)
    return output


def convert_degree_to_km(pts1: np.ndarray, pts2: np.ndarray) -> List[np.ndarray]:
    """Takes two sets of points, each with a lat and lon degree, and computes the distance between each pair in km.
    Note: e.g. pts1 -> np.array([lon, lat])."""
    # https://stackoverflow.com/questions/24617013/convert-latitude-and-longitude-to-x-and-y-grid-system-using-python
    if len(pts1.shape) > 1:
        dx = (pts1[:, 0] - pts2[:, 0]) * 40075.2 * np.cos((pts1[:, 1] + pts2[:, 1]) * np.pi/360)/360
        dy = ((pts1[:, 1] - pts2[:, 1]) * 39806.64)/360
    else:
        dx = (pts1[0] - pts2[0]) * 40075.2 * np.cos((pts1[1] + pts2[1]) * np.pi/360)/360
        dy = ((pts1[1] - pts2[1]) * 39806.64)/360
    return dx, dy


def _convert_degree_to_km(lon: np.ndarray, lat: np.ndarray) -> List[np.ndarray]:
    """Takes two sets of points, each with a lat and lon degree, and computes the distance between each pair in km.
    Note: e.g. pts1 -> np.array([lon, lat])."""
    # https://stackoverflow.com/questions/24617013/convert-latitude-and-longitude-to-x-and-y-grid-system-using-python
    x = lon * 40075.2 * np.cos(lat * np.pi/360)/360
    y = lat * (39806.64/360)
    return x, y
