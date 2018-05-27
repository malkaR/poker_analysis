import os

import pandas as pd
import matplotlib as mpl
mpl.use('PDF')
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from settings import (HOLDEM_FOLDER, HOLDEM_YEAR,
    HOLDEM_MONTHS, PLAYER_COLUMNS, DATA_COLUMNS)

class BaseData:
    """
    Base class to read and process a set of data files
    for poker game data.
    """
    data_folder = None
    months = None
    year = None
    columns = None
    full_path = None
    pickle_path = None
    pickle_file_name = None

    def __init__(self, data_folder, year, months, columns):
        super().__init__()
        self.data_folder = data_folder
        self.months = months
        self.year = year
        self.columns = columns
    
    def pickle_dataframe(self, df, file_name=None, compression=None): # compression doesn't owrk?
        """Pickle and save the dataframe to a file"""
        if df is None or df.empty:
            return
        if not os.path.exists(self.pickle_path):
            os.makedirs(self.pickle_path)
        file_name = file_name or self.pickle_file_name
        df.to_pickle(
            os.path.join(self.pickle_path, file_name), compression=compression)

    def unpickle_dataframe(self, file_name=None, compression=None):
        """Retreive the dataframe from a pickled file"""
        file_name = file_name or self.pickle_file_name
        return pd.read_pickle(
            os.path.join(self.pickle_path, file_name), compression=compression)
    
    def read_files(self):
        raise NotImplemented
        
    def extract_data(self, df):
        """
        Extract the desired data. Subclasses should override this method
        to perform custom aggregation.
        """
        raise NotImplemented

        
class MonthlyData(BaseData):
    """Read and process a set of files for a month's worth of poker data"""
    def __init__(self, data_folder, year, months, columns):
        super().__init__(data_folder, year, months, columns)
        self.full_path = os.path.join(data_folder, year + months)
        self.pickle_path = os.path.join(data_folder, year)
        self.pickle_file_name = months

    def read_files(self):
        """
        Read the files in the monthly directory and
        combine all to one dataframe
        """
        if not os.path.exists(self.full_path):
            return pd.DataFrame(columns=self.columns)
        df_list = []
        for root, _, files in os.walk(self.full_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.exists(file_path):
                    df_temp = pd.read_table(
                        file_path,
                        delim_whitespace=True,
                        header=None,
                        names=self.columns)
                    df_list.append(df_temp)
        return pd.concat(df_list, ignore_index=True)

    def extract_data(self, df):
        """
        Extract data for each player, including number of wins,
        number of games played, and total monetary gains
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=DATA_COLUMNS)
        df['is_win'] = df.pot_winnings_amount.apply(lambda x: x > 0)

        game_count = df.groupby('player_name').count()['game_id'] \
            .rename('game_count')
        monetary_gain = (df.groupby('player_name').pot_winnings_amount.sum()
                        - df.groupby('player_name').pot_input_amount.sum()
                        ).rename('monetary_gain')
        num_wins = df.groupby('player_name').is_win.value_counts()[:, 1].rename('num_wins')
        df_new = pd.concat([game_count, monetary_gain, num_wins], axis=1).fillna(0)
        return df_new


class YearlyData(BaseData):
    """
    Read a set of monthly data from pickled files and
    combine to produce one dataframe
    """
    pickle_file_names = None

    def __init__(self, data_folder, year, months, columns):
        super().__init__(data_folder, year, months, columns)
        self.full_path = os.path.join(data_folder, year)
        self.pickle_path = self.full_path
        self.pickle_file_names = months

    def read_files(self):
        """
        Read the pickled files in the year directory and
        combine all to one dataframe
        """
        df_list = []
        for root, _, files in os.walk(self.full_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.exists(file_path):
                    df_list.append(self.unpickle_dataframe(file_name=file_name))
        return pd.concat(df_list).rename_axis('player_name')
    
    def extract_data(self, df):
        """
        For each player, sum the yearly data to combine the
        results from different months.
        """
        game_count = df.groupby('player_name').game_count.sum()
        monetary_gain = df.groupby('player_name').monetary_gain.sum()
        num_wins = df.groupby('player_name').num_wins.sum()
        return pd.concat([game_count, monetary_gain, num_wins], axis=1)


def apply_to_series(series, func):
    return series.apply(func)

def compute_from_series(series1, series2, func):
    return func(series1, series2)

if __name__ == '__main__':
    for year in HOLDEM_YEAR:
        for month in HOLDEM_MONTHS:
            monthly_data = MonthlyData(HOLDEM_FOLDER, year, month, PLAYER_COLUMNS)
            df = monthly_data.read_files()
            df = monthly_data.extract_data(df)
            monthly_data.pickle_dataframe(df)
    df_list = []
    for year in HOLDEM_YEAR:
        yearly_data = YearlyData(HOLDEM_FOLDER, year, HOLDEM_MONTHS, DATA_COLUMNS)
        df = yearly_data.read_files()
        df_list.append(yearly_data.extract_data(df))
    df = pd.concat(df_list)
    game_count = df.groupby('player_name').game_count.sum()
    monetary_gain = df.groupby('player_name').monetary_gain.sum()
    num_wins = df.groupby('player_name').num_wins.sum()
    df = pd.concat([game_count, monetary_gain, num_wins], axis=1)
        
    with PdfPages('multipage_pdf.pdf') as pdf:
        df.plot()
        pdf.savefig()
        mpl.pyplot.close()

        df.plot(kind='scatter', x='num_wins', y='monetary_gain')
        pdf.savefig()
        mpl.pyplot.close()