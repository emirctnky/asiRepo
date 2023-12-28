"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline
from asi_project.pipelines.data_processing.pipeline import create_pipeline as create_data_processing_pipeline
from asi_project.pipelines.model_training.pipeline import create_pipeline as create_model_training_pipeline
from asi_project.pipelines.incremental_training.pipeline import create_incremental_training_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    data_processing_pipeline = create_data_processing_pipeline()
    model_training_pipeline = create_model_training_pipeline()

    return {
        "data_processing": data_processing_pipeline,
        "model_training": model_training_pipeline,
        "incremental_training": create_incremental_training_pipeline(),
        "__default__": data_processing_pipeline + model_training_pipeline,
    }
