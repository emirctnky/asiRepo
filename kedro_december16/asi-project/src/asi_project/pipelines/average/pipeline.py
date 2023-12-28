"""
This is a boilerplate pipeline 'average'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node
from .nodes import load_data, calculate_average_delivery_time

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(load_data, inputs=None, outputs="raw_data", name="load_data_node"),
            node(calculate_average_delivery_time, inputs="raw_data", outputs="average_delivery_time", name="calculate_average_delivery_time_node"),
        ]
    )

