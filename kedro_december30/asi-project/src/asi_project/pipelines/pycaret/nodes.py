import pandas as pd
from pycaret.classification import setup, compare_models,predict_model
import logging

logger = logging.getLogger(__name__)

def train_pycaret_automl(data: pd.DataFrame, test_data: pd.DataFrame = None) -> dict:
    """
    Trains PyCaret AutoML models, returns the best model, and optionally evaluates it.
    """
    clf_setup = setup(data, target='Time_taken(min)', session_id=123, verbose=False)
    best_model = compare_models(include=['rf','lr','dt'])

    results = {"best_model": best_model}

    # Evaluate model if test_data is provided
    if test_data is not None:
        predictions = predict_model(best_model, data=test_data)
        results["predictions"] = predictions

        # Log performance metrics
        logger.info(f"Best Model: {best_model}")
        logger.info(f"Evaluation Metrics: {predictions}")

    return results
