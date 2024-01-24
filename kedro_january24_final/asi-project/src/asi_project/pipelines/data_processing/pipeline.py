from kedro.pipeline import Pipeline, node
from asi_project.pipelines.data_processing.nodes import (
    read_data,
    check_missing_values,
    compute_distance,
    prepare_features_target,
    data_split,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=read_data,
                inputs="params:deliverytime_data_path",
                outputs="raw_data",
                name="read_data_node"
            ),
            node(
                func=check_missing_values,
                inputs="raw_data",
                outputs="clean_data",
                name="check_missing_values_node"
            ),
            node(
                func=compute_distance,
                inputs="clean_data",
                outputs="distance_data",
                name="compute_distance_node"
            ),
            node(
                func=prepare_features_target,
                inputs="distance_data",
                outputs=["features", "target"],
                name="prepare_features_target_node"
            ),
            node(
                func=data_split,
                inputs=["features", "target", "params:test_size", "params:random_state"],
                outputs=["x_train", "x_test", "y_train", "y_test"],
                name="data_split_node"
            ),
        ]
    )
