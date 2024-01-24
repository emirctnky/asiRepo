from kedro.pipeline import Pipeline, node
from .nodes import train_pycaret_automl
 
def create_pycaret_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_pycaret_automl,
                inputs="deliverytime_data",
                outputs="pycaret_best_model",
                name="train_pycaret_automl_node"
            ),
        ]
    )
