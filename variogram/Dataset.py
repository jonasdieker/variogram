import pandas as pd
import os
from typing import List
import datetime


class Dataset:
    """This class handles the data loading for the Variogram computation.
    """
    def __init__(self, data_path: str):
        """
            data_path (str): Can be either a folder containing csv files or a single
                             csv file
        """

        self.data_path = data_path
        self.is_file = True if data_path.find(".") != -1 else False
        if self.is_file:
            if data_path.split(".")[-1] == "csv":
                files = [data_path]
        else:
            files = os.listdir(data_path)
            self.csv_files = [file for file in files if file.split(".")[-1] == "csv"]
            if self.csv_files:
                files = [os.path.join(data_path, file) for file in self.csv_files]
        self.files = sorted(files)

    def load_all_files(self, overlap: bool=True, verbose: bool=False) -> pd.DataFrame:
        """Loads entire dataset specified.
        """
        if self.is_file:
            raise ValueError("Provided a file not a folder.Please use the load_single_file_by_idx function!")

        df = pd.read_csv(self.files[0])
        df = pd.DataFrame(columns=df.columns)

        for i in range(len(self.files)):
            df_temp = pd.read_csv(self.files[i])
            if overlap is False:
                times = sorted(list(set(df_temp["time"])))[:24]
                df_temp = df_temp[df_temp["time"].isin(times)]
            df = pd.concat([df, df_temp], ignore_index=True)
        print(f"Loaded {self.dataset_name} from {self.dataset_type}.")
        if verbose:
            print(df.describe())
        return df

    def load_single_file_by_idx(self, file_idx: int=0, verbose: bool=False) -> pd.DataFrame:
        """Loads any file of specified dataset.
        """
        if self.is_file:
            pd.read_csv(self.data_path)

        df = pd.read_csv(self.files[file_idx])
        print(f"Loaded: {self.files[file_idx]}")
        if verbose:
            print(df.describe())
        return df

    def load_single_file_by_name(self, file_name: str) -> pd.DataFrame:
        """Loads a specific file of name file_name.
        """
        if self.is_file:
            pd.read_csv(self.data_path)

        if file_name not in self.csv_files:
            raise FileExistsError(f"Specified file '{file_name}' does not exist!")
        df = pd.read_csv(os.path.join(self.data_path, file_name))
        return df
