import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import numpy as np


class FeaturePreparation(object):

    @staticmethod
    def separate_num_cat_target(df, is_validation=False):
        cat = [col for col in df.columns if df.dtypes[col] == "object"]
        target = ["attractiveness"]
        num = [col for col in df.columns if col not in cat + target + ["id"]]

        df_num = df[num]
        df_cat = df[cat]

        df_cat = pd.get_dummies(df_cat, prefix=cat)

        X = pd.concat([df_num, df_cat], axis=1)
        if not is_validation:
            y = df[target[0]]
            return X, y
        else:
            return X

    @staticmethod
    def null_imputer(df, values_to_input, n_neighbors=10):
        cols = [col for col in df.columns if col != "prod_cost"]
        imputer = KNeighborsRegressor(n_neighbors)
        return imputer.fit(df[~values_to_input][cols], df[~values_to_input]["prod_cost"])

    def input_knn(self, df, target_vector, values_to_input, n_neighbors=8):
        null_predictor = self.null_imputer(df, values_to_input, n_neighbors)
        cols = [col for col in df.columns if col != "prod_cost"]
        result = np.zeros(len(target_vector))
        for i in target_vector.index:
            if i in list(df[values_to_input].index):
                value_to_predict = df[values_to_input][cols].loc[i, :].values.reshape(1, -1)
                result[i] = null_predictor.predict(value_to_predict)[0]
            else:
                result[i] = target_vector.loc[i, "prod_cost"]
        return result
