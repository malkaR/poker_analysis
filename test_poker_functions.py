import operator
import os
import unittest

import numpy as np
import pandas as pd

from poker_functions import (
    MonthlyData, YearlyData)
from settings import DATA_COLUMNS, PLAYER_COLUMNS

# TODO: check for non existent files, bad input data and other edge cases


class TestProcessGameData(unittest.TestCase):
    TEST_FOLDER = '/home/malkacod/projects/poker_code/poker_code/test_data'
    TEST_YEARS = '1995 1996'.split()
    TEST_MONTHS = '04 05'.split()

    def setUp(self):
        self.TEST_YEAR = self.TEST_YEARS[0]
        self.monthly_data = MonthlyData(
            self.TEST_FOLDER, self.TEST_YEAR, '04', PLAYER_COLUMNS)
        self.monthly_data_two = MonthlyData(
            self.TEST_FOLDER, self.TEST_YEAR, '04', PLAYER_COLUMNS)
        self.player_data = {
            'player_name': ['deadhead', 'justnuts', 'rimedio'],
            'game_count': [4, 4, 6],
            'monetary_gain': [125, 0, -180],
            'num_wins': [2, 0, 1]
            }

    def tearDown(self):
        for year in self.TEST_YEARS:
            test_path = os.path.join(self.TEST_FOLDER, year)
            for root, _, files in os.walk(test_path):
                for file in files:
                    if os.path.exists(os.path.join(root, file)):
                        os.remove(os.path.join(root, file))

    def test_read_month_data(self):
        df = self.monthly_data.read_files()
        self.assertEqual(df.shape[0], 14)
        # TODO assert contents of data

    def test_pickle_unpickle_paths(self):
        df1 = self.monthly_data.read_files()
        self.monthly_data.pickle_dataframe(df1)
        df2 = self.monthly_data.unpickle_dataframe()
        pd.testing.assert_frame_equal(df1, df2)

    def test_extract_month_data(self):
        df = self.monthly_data.read_files()
        result = self.monthly_data.extract_data(df)
        expected_result = pd.DataFrame(
            self.player_data).set_index('player_name')
        pd.testing.assert_frame_equal(result, expected_result,
                                      check_like=True, check_dtype=False)

    def test_run_multiple_months(self):
        MonthlyData.run_multiple_months(
            self.TEST_FOLDER, self.TEST_YEAR, self.TEST_MONTHS, PLAYER_COLUMNS)
        for month in self.TEST_MONTHS:
            self.assertTrue(os.path.exists(
                os.path.join(
                    self.TEST_FOLDER, os.path.join(self.TEST_YEAR, month))))
        # TODO: check contents of pickled files can be read
        # back to a dataframe

    def setup_year_data(self):
        df = self.monthly_data.read_files()
        df = self.monthly_data.extract_data(df)
        self.monthly_data.pickle_dataframe(df, file_name='001')
        df = self.monthly_data_two.read_files()
        df = self.monthly_data_two.extract_data(df)
        self.monthly_data_two.pickle_dataframe(df, file_name='002')
        yearly_data = YearlyData(
            self.TEST_FOLDER, self.TEST_YEAR, ['001', '002'],
            DATA_COLUMNS)
        return yearly_data

    def test_read_year_data(self):
        yearly_data = self.setup_year_data()
        result = yearly_data.read_files()
        player_data_twice = {key: val * 2 for key, val in
                             self.player_data.items()}
        expected_result = pd.DataFrame(
            player_data_twice).set_index('player_name')
        pd.testing.assert_frame_equal(result, expected_result,
                                      check_like=True, check_dtype=False)

    def test_extract_year_data(self):
        yearly_data = self.setup_year_data()
        df = yearly_data.read_files()
        result = yearly_data.extract_data(df)
        player_data_doubled = {key: [item if isinstance(item, str) else
                                     item * 2 for item in val] for
                               key, val in self.player_data.items()}
        expected_result = pd.DataFrame(
            player_data_doubled).set_index('player_name')
        pd.testing.assert_frame_equal(result, expected_result,
                                      check_like=True, check_dtype=False)

    def test_run_multiple_years(self):
        for year in self.TEST_YEARS:
            MonthlyData.run_multiple_months(
                self.TEST_FOLDER, year, self.TEST_MONTHS, PLAYER_COLUMNS)
        result = YearlyData.run_multiple_years(
            self.TEST_FOLDER, self.TEST_YEARS, self.TEST_MONTHS, DATA_COLUMNS)
        player_data_quadrupled = {key: [item if isinstance(item, str) else
                                        item * 4 for item in val] for
                                  key, val in self.player_data.items()}
        expected_result = pd.DataFrame(
            player_data_quadrupled).set_index('player_name')
        pd.testing.assert_frame_equal(result, expected_result,
                                      check_like=True, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
