from typing import Tuple, Dict
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from custom_package.dist_utils import distcalculate as compute_distance

def preprocess_additional_data(additional_data: pd.DataFrame, test_size: float, random_state: int):
    additional_data_cleaned = additional_data.dropna()

    additional_data_cleaned['distance'] = additional_data_cleaned.apply(
        lambda row: compute_distance(row['Restaurant_latitude'], row['Restaurant_longitude'], 
                                     row['Delivery_location_latitude'], row['Delivery_location_longitude']),
        axis=1
    )

    X = additional_data_cleaned[["Delivery_person_Age", "Delivery_person_Ratings", "distance"]]
    y = additional_data_cleaned[["Time_taken(min)"]]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return x_train, x_test, y_train, y_test


def incremental_train_rf(x_train_additional: pd.DataFrame, y_train_additional: pd.Series, rf_model_path: str) -> RandomForestRegressor:
    with open(rf_model_path, 'rb') as file:
        rf_model = pickle.load(file)

    rf_model.fit(x_train_additional, y_train_additional.values.ravel())
    return rf_model

def incremental_train_gb(x_train_additional: pd.DataFrame, y_train_additional: pd.Series, gb_model_path: str) -> GradientBoostingRegressor:
    with open(gb_model_path, 'rb') as file:
        gb_model = pickle.load(file)

    gb_model.fit(x_train_additional, y_train_additional.values.ravel())
    return gb_model
