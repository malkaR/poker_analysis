import argparse
import logging
import multiprocessing
import os
import time

import pandas as pd
import matplotlib as mpl
mpl.use('PDF')
from matplotlib.backends.backend_pdf import PdfPages  # noqa
import numpy as np  # noqa

from settings import (DATA_COLUMNS, DATA_FILE_PREFIX, HOLDEM_FOLDER,
                      HOLDEM_MONTHS, HOLDEM_YEARS, PLAYER_COLUMNS)  # noqa


_log = logging.getLogger(__name__)


class PickleDataFrame:
    """
    Base class to read and process a set of data files
    for poker game data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pickle_path = kwargs.get('pickle_path', '')
        self.pickle_file_name = kwargs.get('pickle_file_name', '')

    def pickle_dataframe(self, df, file_name=None):
        """Pickle and save the dataframe to a file"""
        if df is None or df.empty:
            return
        if not os.path.exists(self.pickle_path):
            os.makedirs(self.pickle_path)
        file_name = file_name or self.pickle_file_name
        df.to_pickle(
            os.path.join(self.pickle_path, file_name))

    def unpickle_dataframe(self, file_name=None):
        """Retreive the dataframe from a pickled file"""
        file_name = file_name or self.pickle_file_name
        return pd.read_pickle(
            os.path.join(self.pickle_path, file_name))

    def read_files(self):
        raise NotImplementedError

    def extract_data(self, df):
        """
        Extract the desired data. Subclasses should override this method
        to perform custom aggregation.
        """
        raise NotImplementedError


class MonthlyData(PickleDataFrame):
    """Read and process a set of files for a month's worth of poker data"""
    def __init__(self, data_folder, year, month, columns):
        self.read_path = os.path.join(data_folder, year + month)
        self.columns = columns
        super().__init__(pickle_path=os.path.join(data_folder, year),
                         pickle_file_name=month)

    def read_files(self):
        """
        Read the files in the monthly directory and
        combine all to one dataframe
        """
        df_list = []
        for root, _, files in os.walk(self.read_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if (file_name[0:4] == DATA_FILE_PREFIX and
                        os.path.exists(file_path)):
                    _log.info('working on file {0}'.format(file_name))
                    df_temp = pd.read_table(
                        file_path,
                        delim_whitespace=True,
                        header=None,
                        names=self.columns)
                    df_list.append(df_temp)
        if df_list:
            return pd.concat(df_list)
        return pd.DataFrame(columns=self.columns)

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
        monetary_gain = (df.groupby('player_name').pot_winnings_amount.sum() -
                         df.groupby('player_name').pot_input_amount.sum()
                         ).rename('monetary_gain')
        num_wins = df.groupby('player_name').is_win.value_counts()[:, 1] \
            .rename('num_wins')
        df_new = pd.concat([
            game_count, monetary_gain, num_wins], axis=1).fillna(0)
        return df_new

    @classmethod
    def run_multiple_months(cls, data_folder, year, months, columns):
        """
        Read and process multiple months for a year's worth of
        poker data, then pickle and store each month's result.
        """
        for month in months:
            _log.info('working on year {0} and month {1}'.format(year, month))
            monthly_data = cls(data_folder, year, month, columns)
            df = monthly_data.read_files()
            df = monthly_data.extract_data(df)
            monthly_data.pickle_dataframe(df)


class YearlyData(PickleDataFrame):
    """
    Read a set of monthly data from pickled files and
    combine to produce one dataframe
    """
    def __init__(self, data_folder, year, months, columns):
        self.read_path = os.path.join(data_folder, year)
        super().__init__(pickle_path=self.read_path,
                         pickle_file_name=months)

    def read_files(self):
        """
        Read the pickled files in the year directory and
        combine all to one dataframe
        """
        df_list = []
        for root, _, files in os.walk(self.read_path):
            for file_name in files:
                if file_name not in self.pickle_file_name:
                    continue
                file_path = os.path.join(root, file_name)
                if os.path.exists(file_path):
                    df_list.append(
                        self.unpickle_dataframe(file_name=file_name))
        if df_list:
            return pd.concat(df_list).rename_axis('player_name')
        return pd.DataFrame(columns=self.columns)

    def extract_data(self, df):
        """
        For each player, sum the yearly data to combine the
        results from different months.
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=PLAYER_COLUMNS)
        game_count = df.groupby('player_name').game_count.sum()
        monetary_gain = df.groupby('player_name').monetary_gain.sum()
        num_wins = df.groupby('player_name').num_wins.sum()
        return pd.concat([game_count, monetary_gain, num_wins], axis=1)

    @classmethod
    def run_multiple_years(cls, data_folder, years, months, columns):
        """
        Read and process pickled files storing multiple months and years
        of poker data.
        """
        df_list = []
        for year in years:
            _log.info('working on year {0}'.format(year))
            yearly_data = cls(data_folder, year, months, columns)
            df = yearly_data.read_files()
            df = yearly_data.extract_data(df)
            if not df.empty:
                df_list.append(df)
        if not df_list:
            return pd.DataFrame(columns=columns)
        df = pd.concat(df_list)
        game_count = df.groupby('player_name').game_count.sum()
        monetary_gain = df.groupby('player_name').monetary_gain.sum()
        num_wins = df.groupby('player_name').num_wins.sum()
        df = pd.concat([game_count, monetary_gain, num_wins], axis=1)
        return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze poker data files.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        default=False,
                        help='output logging statements')
    parser.add_argument('-t', '--time-it', dest='time_it', action='store_true',
                        default=False,
                        help='compute run time')
    parser.add_argument('-p', '--processes', dest='number_of_procs',
                        type=int, default=1,
                        help='number of processes to use')
    parser.add_argument('-g', '--graph-only', dest='graph_only',
                        action='store_true', default=False,
                        help='aggregate data to generate a graph from previously saved data')
    options = parser.parse_args()

    if options.verbose:
        _log.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        _log.addHandler(handler)

    if options.time_it:
        start = time.time()

    if not options.graph_only:
        if options.number_of_procs > 1:
            num = options.number_of_procs
            with multiprocessing.Pool(processes=num) as pool:
                results = []
                for year in HOLDEM_YEARS:
                    results.append(pool.apply_async(
                        MonthlyData.run_multiple_months,
                        (HOLDEM_FOLDER, year, HOLDEM_MONTHS, PLAYER_COLUMNS)))
                    for res in results:
                        res.get()
        else:
            for year in HOLDEM_YEARS:
                MonthlyData.run_multiple_months(
                    HOLDEM_FOLDER, year, HOLDEM_MONTHS, PLAYER_COLUMNS)

    df = YearlyData.run_multiple_years(
        HOLDEM_FOLDER, HOLDEM_YEARS, HOLDEM_MONTHS, DATA_COLUMNS)

    if options.time_it:
        end = time.time()
        _log.info('elapsed time: {0}'.format(end - start))

    with PdfPages('multipage_pdf.pdf') as pdf:
        df_largest = df.nlargest(40, 'monetary_gain')
        df_largest.plot()
        pdf.savefig()
        mpl.pyplot.close()

        df_largest.plot(kind='scatter', x='num_wins', y='monetary_gain')
        pdf.savefig()
        mpl.pyplot.close()

        df_largest.plot(kind='scatter', x='game_count', y='num_wins')
        pdf.savefig()
        mpl.pyplot.close()
