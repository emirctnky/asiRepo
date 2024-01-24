from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

def objective_rf_inner(trial, x_train, y_train, x_test, y_test):

    n_estimators = trial.suggest_int("n_estimators", 10, 100)
    max_depth = trial.suggest_int("max_depth", 2, 32)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return mean_squared_error(y_test, preds, squared=False)

def objective_rf(x_train, y_train, x_test, y_test):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_rf_inner(trial, x_train, y_train, x_test, y_test), n_trials=10)
    return study.best_trial.params

def objective_gb_inner(trial, x_train, y_train, x_test, y_test):
    n_estimators = trial.suggest_int("n_estimators", 10, 100)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5)
    max_depth = trial.suggest_int("max_depth", 2, 32)
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return mean_squared_error(y_test, preds, squared=False)

def objective_gb(x_train, y_train, x_test, y_test):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_gb_inner(trial, x_train, y_train, x_test, y_test), n_trials=10)
    return study.best_trial.params


def objective_lstm_inner(trial, x_train, y_train, x_test, y_test):
    # Optuna suggests the number of units for each layer and training parameters
    lstm_units_1 = trial.suggest_categorical("lstm_units_1", [64, 128])
    lstm_units_2 = trial.suggest_categorical("lstm_units_2", [32, 64])
    dense_units = trial.suggest_categorical("dense_units", [10, 25])
    batch_size = trial.suggest_categorical("batch_size", [1, 2])
    epochs = trial.suggest_int("epochs", 1,10)

    # Model definition
    model = Sequential()
    model.add(LSTM(lstm_units_1, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(lstm_units_2, return_sequences=False))
    model.add(Dense(dense_units))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Model training
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    # Evaluate the model
    preds = model.predict(x_test)
    return mean_squared_error(y_test, preds, squared=False)

def objective_lstm(x_train, y_train, x_test, y_test):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_lstm_inner(trial, x_train, y_train, x_test, y_test), n_trials=10)
    return study.best_trial.params
