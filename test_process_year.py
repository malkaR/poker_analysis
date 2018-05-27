import operator
import os
import unittest

import numpy as np
import pandas as pd

from functions import (
    MonthlyData, YearlyData, apply_to_series, compute_from_series)
from settings import PLAYER_COLUMNS

# cleanup created files
# multiply by 2 in dict comprehenstion
# separate in to two classes
# check for non existent files, bad input data and other edge cases

class TestProcessGameData(unittest.TestCase):
    TEST_FOLDER = '/home/malkacod/projects/poker_code/poker_code/test_data'
    TEST_YEAR = '1995'    
    # TODO test code to clean up after itself and delete created files
    
    def setUp(self):
        self.series = pd.Series({'a':1, 'b':2, 'c':3, 'd':0})
        self.series2 = pd.Series({'a':3, 'b':1, 'c':0, 'd':-4})
        self.series3 = pd.Series({'a':[1, 2], 'b':[4,6], 'c':[0,3,9]})
        
        self.monthly_data = MonthlyData(
            self.TEST_FOLDER, self.TEST_YEAR, '04', PLAYER_COLUMNS)
        self.monthly_data_two = MonthlyData(
            self.TEST_FOLDER, self.TEST_YEAR, '04', PLAYER_COLUMNS)
    
    def tearDown(self):
        test_path = os.path.join(self.TEST_FOLDER, self.TEST_YEAR)
        for root, _, files in os.walk(test_path):
            for file in files:
                os.remove(os.path.join(root, file))

    def test_apply_to_series_gt_zero(self):
        result = apply_to_series(self.series, lambda x: x > 0)
        expected_result = pd.Series(
            {'a': True, 'b': True, 'c':True, 'd':False})
        pd.testing.assert_series_equal(result, expected_result)
    
    def test_apply_to_series_mean(self):
        result = apply_to_series(self.series3, np.mean)
        expected_result = pd.Series({'a':1.5, 'b':5, 'c':4})
        pd.testing.assert_series_equal(result, expected_result)
        
    def test_compute_from_series_subtraction(self):
        result = compute_from_series(self.series, self.series2, operator.sub)
        expected_result = pd.Series(
            {'a': -2, 'b': 1, 'c':3, 'd':4})
        pd.testing.assert_series_equal(result, expected_result)
    
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
            {'player_name':['deadhead', 'rimedio', 'justnuts'],
             'game_count': [4, 6, 4],
             'monetary_gain':[125, -180, 0],
             'num_wins':[2, 1, 0]
            }).set_index('player_name')
        pd.testing.assert_frame_equal(result, expected_result,
                                      check_like=True, check_dtype=False)
    
    def setup_year_data(self):
        df = self.monthly_data.read_files()
        df = self.monthly_data.extract_data(df)
        self.monthly_data.pickle_dataframe(df, file_name='001')
        df = self.monthly_data_two.read_files()
        df = self.monthly_data_two.extract_data(df)
        self.monthly_data_two.pickle_dataframe(df, file_name='002')
        yearly_data = YearlyData(self.TEST_FOLDER, self.TEST_YEAR, ['001', '002'],
                   ['player_name', 'game_count', 'monetary_gain', 'num_wins'])
        return yearly_data

    def test_read_year_data(self):
        yearly_data = self.setup_year_data()
        result = yearly_data.read_files()
        expected_result = pd.DataFrame(
            {'player_name': ['deadhead', 'justnuts', 'rimedio', 'deadhead', 'justnuts', 'rimedio'],
             'game_count': [4, 4, 6, 4, 4, 6],
             'monetary_gain':[125, 0, -180, 125, 0, -180],
             'num_wins':[2, 0, 1, 2, 0, 1]
            }).set_index('player_name')
        pd.testing.assert_frame_equal(result, expected_result,
                                      check_like=True, check_dtype=False)

    def test_extract_year_data(self):
        yearly_data = self.setup_year_data()
        df = yearly_data.read_files()
        result = yearly_data.extract_data(df)
        expected_result = pd.DataFrame(
            {'player_name':['deadhead', 'rimedio', 'justnuts'],
             'game_count': [4*2, 6*2, 4*2],
             'monetary_gain':[125*2, -180*2, 0],
             'num_wins':[2*2 ,1*2, 0]
            }).set_index('player_name')
        pd.testing.assert_frame_equal(result, expected_result,
                                      check_like=True, check_dtype=False)
                   

if __name__ == '__main__':
    unittest.main()
