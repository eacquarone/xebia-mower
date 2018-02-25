
class DataFrameCleaner(object):
    @staticmethod
    def clean_warranty(warranty):
        year = warranty[0]
        if year == "1":
            return "1 an"
        elif year in ["2", "3"]:
            return "%s ans"%(year)
        else:
            raise ValueError("This value is not recognized")

    @staticmethod
    def drop_columns(df, is_validation=False):
        df = df.drop("product_type", axis=1)
        if is_validation:
            return df
        else:
            return df.drop("market_share", axis=1)
