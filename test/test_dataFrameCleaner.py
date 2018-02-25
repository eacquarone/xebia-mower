from unittest import TestCase
from utils.clean_dataframe import DataFrameCleaner
import pandas as pd
import pandas.testing as pdt

class TestDataFrameCleaner(TestCase):
    def test_clean_warranty(self):
        cleaner = DataFrameCleaner()
        test_cases = ["1_ans", "2 ans.", "Xfe"]
        self.assertEqual(cleaner.clean_warranty(test_cases[0]), "1 an")
        self.assertEqual(cleaner.clean_warranty(test_cases[1]), "2 ans")
        with self.assertRaises(ValueError):
            cleaner.clean_warranty(test_cases[2])

    def test_drop_columns(self):
        cleaner = DataFrameCleaner()
        test_cases = [
            pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=["prod_cost", "product_type"]),
            pd.DataFrame([[1, 2, 5], [3, 4, 6], [5, 6, 2]], columns=["prod_cost", "product_type", "market_share"])
        ]
        expected_result = pd.DataFrame([[1], [3], [5]], columns=["prod_cost"])
        pdt.assert_frame_equal(cleaner.drop_columns(test_cases[0], True), expected_result)
        pdt.assert_frame_equal(cleaner.drop_columns(test_cases[1]), expected_result)
