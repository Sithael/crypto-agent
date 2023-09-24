import random
import re
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from pandas import DataFrame

from scraper.utils.common import constants


class DataIterator:
    """
    Offline Data Loader [ csv file ]

    Divides existing dataset into smaller chunks of time periods
    Component structured in a iterable way so to ensure common usage with step method

    Args:
        :param [crypto_type]: crypto dataset used to train the agent [ btc, ether ]
        :param [time_interval]: value depicting the evaluation time for the agent
        :param [black_size]: number of adjacent time intervals
    """

    def __init__(self, crypto_type: str, time_interval: str, day_count: int) -> None:
        self._current_index = 0
        self.crypto_type = crypto_type
        self.time_interval = time_interval
        self.day_count = day_count
        self.data_directory = self._fetch_data_directory()
        self.data = self._preprocess_data(self.data_directory)

    def _fetch_data_directory(self) -> Path:
        """Get crypto data directory"""
        import sys, os

        current_file_dir = os.path.abspath(
            sys.modules[DataIterator.__module__].__file__
        )
        target_dir = current_file_dir.split(os.sep)[:-3]
        path_to_scraper = Path(os.path.join(*target_dir))
        if self.crypto_type == "btc":
            return (
                Path("/")
                / path_to_scraper
                / constants.DATA_DIR
                / constants.OfflineDataStream.BITCOIN_DATA_FNAME.value
            )
        else:
            return None

    def _preprocess_data(self, data_path: Path) -> DataFrame:
        """open and preprocess csv file"""
        crypto_df = pd.read_csv(data_path)
        crypto_interpolated_df = crypto_df[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume_(BTC)",
                "Volume_(Currency)",
                "Weighted_Price",
            ]
        ].interpolate()
        crypto_interpolated_df["Timestamp"] = crypto_df["Timestamp"]

        crypto_interpolated_df["Date"] = pd.to_datetime(
            crypto_interpolated_df["Timestamp"], unit="s"
        )
        crypto_interpolated_df.set_index("Timestamp", inplace=True)
        crypto_interpolated_df["Year"] = crypto_interpolated_df["Date"].dt.year

        crypto_correct_data_df = crypto_interpolated_df[
            ~(crypto_interpolated_df["Date"] < "2018-01-01")
        ]
        return crypto_correct_data_df

    def _hit_data_tail(self, start: int, block_step: int, max_index: int) -> bool:
        """check wheter there is no jump to the past once computing slice"""
        overlap = (start + block_step) / max_index
        if overlap > 1.0:
            return True
        else:
            return False

    def _compute_window_slide(self):
        """compute slice properties based on parameters in ctor"""
        int_time_interval = int(re.search(r"\d+", self.time_interval).group())
        minutes_x_freq = 1440 / int_time_interval
        window_slide = int(minutes_x_freq * self.day_count)
        return window_slide

    def _normalize_data_window(self, dataframe: DataFrame) -> DataFrame:
        """normalize dataframe"""
        normalized_df = dataframe.copy()
        column_names = normalized_df.columns.tolist()
        for column in column_names:
            if column == "Date" or column == "Year":
                raise TypeError(
                    "Could not normalize Year and Data series - consider dropping before normalizing"
                )
            else:
                next_shift = normalized_df[column].shift(1)
                normalized_df[column] = np.log(normalized_df[column]) - np.log(
                    next_shift
                )
                min_col = normalized_df[column].min()
                max_col = normalized_df[column].max()
                normalized_df[column] = (normalized_df[column] - min_col) / (
                    max_col - min_col
                )
        clean_normalized_df = normalized_df.dropna()
        is_nan = clean_normalized_df.isna().sum()
        if is_nan.any():
            raise SystemError("NaN value found")
        return clean_normalized_df

    def _prepare_data_time_window(self) -> Union[DataFrame, DataFrame]:
        """extract time window from data"""
        start_date = self.data["Date"].min()
        end_date = self.data["Date"].max()
        dates_wrt_frequences = pd.Series(
            pd.date_range(start=start_date, end=end_date, freq=self.time_interval)
        )

        window_slide = self._compute_window_slide()

        selection_start_index = None
        while True:
            selection_start_index = random.choice(dates_wrt_frequences.index)
            is_overlapping = self._hit_data_tail(
                selection_start_index, window_slide, dates_wrt_frequences.index.stop
            )
            if not is_overlapping:
                break

        selection_start_date = dates_wrt_frequences.iloc[selection_start_index]

        """
        selected_dates = dates_wrt_frequences[
            selection_start_index : selection_start_index + window_slide
        ]
        """
        selected_dates = dates_wrt_frequences[selection_start_index::]
        samples = self.data[self.data.Date.isin(selected_dates)]
        target_samples = samples.drop(["Date", "Year"], axis=1)
        normalized_samples = self._normalize_data_window(target_samples)
        return target_samples, normalized_samples, samples

    @property
    def mean_window_crypto_price(self) -> np.ndarray:
        """calculate mean crypto price per window"""
        return self.crypto_window_mean

    def __iter__(self):
        """Returns random time period defined in DataIterator ctor"""
        samples, norm_samples, orig_samples = self._prepare_data_time_window()
        self.crypto_window_mean = samples["Weighted_Price"].mean()
        self.samples = samples.to_numpy()
        self.norm_samples = norm_samples.to_numpy()
        self.orig_samples = orig_samples.to_numpy()
        return self

    def __next__(self):
        """Return next observation based on the previous timestep"""
        try:
            sample_entity = self.samples[self._current_index]
            normalized_sample_entity = self.norm_samples[self._current_index]
            weighted_price = sample_entity[-1].astype(np.float32)
            self._current_index += 1
            observation = [sample_entity, normalized_sample_entity, weighted_price]
        except IndexError:
            self._current_index = 0
            stop_iteration_flag = constants.OfflineDataStream.STOP_ITERATION_FLAG.value
            return stop_iteration_flag
        except Exception as e:
            raise SystemError(f"System has encountered unknown malfunction: {e}")
        return observation
