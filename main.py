import pandas as pd
from utils.clean_dataframe import DataFrameCleaner
from utils.prepare_features import FeaturePreparation
from utils.metrics import Metrics

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

import numpy as np

def main():
    rmsle = Metrics().rmsle
    dataframe_cleaner = DataFrameCleaner()
    prepare_features = FeaturePreparation()

    validation_set = pd.read_csv("./Data/submission_set.csv", sep=";")
    df = pd.read_csv("./Data/mower_market_snapshot.csv", sep=";")

    validation_set = dataframe_cleaner.drop_columns(validation_set, True)
    df = dataframe_cleaner.drop_columns(df)

    input_value = 0.

    df["prod_cost"] = df["prod_cost"].apply(lambda s: float(s) if s != "unknown" else input_value)
    df["warranty"] = df["warranty"].apply(lambda w: dataframe_cleaner.clean_warranty(w))

    X_val = prepare_features.separate_num_cat_target(validation_set, True)
    X, y = prepare_features.separate_num_cat_target(df)

    input_knn = prepare_features.input_knn

    X_k = X

    null_prod_costs = (df.prod_cost.isna()) | (df.prod_cost <= 0)
    X_k["prod_cost"] = input_knn(X_k, X_k[["prod_cost"]], null_prod_costs, 8)

    X_k_train, X_k_test, y_k_train, y_k_test = train_test_split(X_k, y, test_size=.3, random_state=1000)

    scaler = StandardScaler()
    pol_feature = PolynomialFeatures(2)

    g_reg = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=.05)

    pipe_reg = Pipeline([("standardizer", scaler),
                         ("polynomial_feature_creation", pol_feature),
                         ("gradient_boosted_tree", g_reg)])
    pipe_reg.fit(X_k_train, np.log(1 + y_k_train))

    print rmsle(np.exp(pipe_reg.predict(X_k_test.values)) - 1, y_k_test)

    validation_set["attractiveness"] = np.exp(pipe_reg.predict(X_val.values)) - 1

    (validation_set[["id", "attractiveness"]]
        .set_index("id")
        .to_csv("./predictions/acquarone_enguerand_attractiveness.csv", sep=";")
    )


if __name__ == '__main__':
    main()