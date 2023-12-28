"""
This is a boilerplate pipeline 'average'
generated using Kedro 0.18.14
"""
from kedro.pipeline.node import node 
import pandas as pd
from typing import Dict, Any
from kedro.framework.hooks import _create_hook_manager

def load_data() -> pd.DataFrame:
    
    file_path = "data/01_raw/deliverytime.csv"  
    return pd.read_csv(file_path)

def calculate_average_delivery_time(data: pd.DataFrame) -> pd.DataFrame:
    average_delivery_time = data['Time_taken(min)'].mean()
    result = pd.DataFrame({'average_delivery_time': [average_delivery_time]})
    return result
