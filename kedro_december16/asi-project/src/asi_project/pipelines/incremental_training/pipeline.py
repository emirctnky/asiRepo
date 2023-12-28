from kedro.pipeline import Pipeline, node
from .nodes import preprocess_additional_data, incremental_train_rf, incremental_train_gb

def create_incremental_training_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocess_additional_data,
                inputs=["additional_data", "params:test_size_additional", "params:random_state_additional"],
                outputs=["x_train_additional", "x_test_additional", "y_train_additional", "y_test_additional"],
                name="preprocess_additional_data_node"
            ),
             node(
                func=incremental_train_rf,
                inputs=["x_train_additional", "y_train_additional", "params:rf_model_path"],
                outputs="rf_model_updated",
                name="incremental_train_rf_node"
            ),
            node(
                func=incremental_train_gb,
                inputs=["x_train_additional", "y_train_additional", "params:gb_model_path"],
                outputs="gb_model_updated",
                name="incremental_train_gb_node"
            ),
        ]
    )
