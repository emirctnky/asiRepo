from kedro.pipeline import Pipeline, node
from .nodes import train_random_forest_model, train_gradient_boosting_model

from asi_project.pipelines.model_training.nodes import (
    scale_features,
    train_model,
    make_prediction
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=scale_features,
                inputs=["x_train", "x_test"],
                outputs=["X_train_scaled", "X_test_scaled"],
                name="scale_features_node"
            ),
            node(
                func=train_model,
                inputs=["X_train_scaled", "y_train"],
                outputs="lstm_model",
                name="train_model_node"
            ),
            node(
                func=train_random_forest_model,
                inputs=["x_train", "y_train"],
                outputs="rf_trained_model",
                name="train_rf_model_node"
            ),
            node(
                func=train_gradient_boosting_model,
                inputs=["x_train", "y_train"],
                outputs="gb_trained_model",
                name="train_gb_model_node"
            ),
            node(
                func=make_prediction,
                inputs=["lstm_model", "X_test_scaled"],
                outputs="predictions",
                name="make_prediction_node"
            ),
        ]
    )
