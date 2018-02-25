from unittest import TestCase
import pandas as pd
import pandas.testing as pdt
import numpy as np
import numpy.testing as npt
from utils.prepare_features import FeaturePreparation


class TestFeaturePreparation(TestCase):
    def test_separate_num_cat_target(self):
        prepare_features = FeaturePreparation()

        test_case = pd.DataFrame([[1334, "Low", 456, .45]], columns=["id", "quality", "price", "attractiveness"])

        expected_result = (pd.DataFrame([[456, 1]], columns=["price", "quality_Low"], dtype=np.int64),
                           pd.DataFrame([.45], columns=["attractiveness"])
                           )
        # We are force to coerce type since pd.get_dummies automatically set type to unsigned int 8 bits
        expected_result[0]["quality_Low"] = expected_result[0]["quality_Low"].astype(np.uint8)
        pdt.assert_frame_equal(prepare_features.separate_num_cat_target(test_case)[0], expected_result[0])
        pdt.assert_series_equal(prepare_features.separate_num_cat_target(test_case)[1],
                                expected_result[1]["attractiveness"])

    def test_null_imputer(self):
        prepare_features = FeaturePreparation()
        null_imputer = prepare_features.null_imputer

        test_case = pd.DataFrame([[1, 1], [1, np.nan]], columns=["col", "prod_cost"])
        values_to_input = test_case.prod_cost.isna()
        expected_result = np.array([1])
        prediction = null_imputer(test_case, values_to_input, 1).predict(test_case[values_to_input][["col"]])
        npt.assert_array_equal(expected_result, prediction)

    def test_input_knn(self):
        prepare_features = FeaturePreparation()
        input_knn = prepare_features.input_knn
        test_case = pd.DataFrame([[1, 1], [1, np.nan]], columns=["col", "prod_cost"])
        values_to_input = test_case.prod_cost.isna()
        expected_result = pd.DataFrame([[1, 1], [1, 1]], columns=["col", "prod_cost"])

        prediction = input_knn(test_case, test_case[["prod_cost"]], values_to_input, 1)

        npt.assert_array_equal(expected_result["prod_cost"].values, prediction)
