from kedro.pipeline import Pipeline, node
from .nodes import objective_rf, objective_gb, objective_lstm

def create_optuna_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=objective_rf,
                inputs=["x_train", "y_train", "x_test", "y_test"],
                outputs="rf_results",
                name="objective_rf_node",
            ),
            node(
                func=objective_gb,
                inputs=["x_train", "y_train", "x_test", "y_test"],
                outputs="gb_results",
                name="objective_gb_node",
            ),
            node(
                func=objective_lstm,
                inputs=["x_train", "y_train", "x_test", "y_test"],
                outputs="lstm_results",
                name="objective_lstm_node",
            ),

        ]
    )
