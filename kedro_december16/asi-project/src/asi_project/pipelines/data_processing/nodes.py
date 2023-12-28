import os
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

from custom_package.dist_utils import distcalculate

def read_data(data_path: str) -> pd.DataFrame:
    print(f"Current working directory: {os.getcwd()}")
    absolute_path = os.path.abspath(data_path)
    print(f"Absolute path of the data file: {absolute_path}")
    return pd.read_csv(data_path)

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()  

def compute_distance(df: pd.DataFrame) -> pd.DataFrame:
    df['distance'] = df.apply(lambda row: distcalculate(row['Restaurant_latitude'], 
                                                        row['Restaurant_longitude'], 
                                                        row['Delivery_location_latitude'], 
                                                        row['Delivery_location_longitude']), axis=1)
    return df

def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = df[["Delivery_person_Age", "Delivery_person_Ratings", "distance"]]
    y = df[["Time_taken(min)"]]
    return X, y

def data_split(X: pd.DataFrame, y: pd.DataFrame, test_size: float, random_state: int):
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test




