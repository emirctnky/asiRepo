from typing import Dict, Tuple
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled
def train_model(X_train_scaled: np.ndarray, y_train: np.ndarray) -> Sequential:
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_scaled, y_train, batch_size=1, epochs=9)
    return model


def train_random_forest_model(x_train: pd.DataFrame, y_train: pd.DataFrame) -> RandomForestRegressor:
    y_train_reshaped = y_train.values.ravel()  
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train_reshaped)
    return rf_model

def train_gradient_boosting_model(x_train: pd.DataFrame, y_train: pd.DataFrame) -> GradientBoostingRegressor:
    y_train_reshaped = y_train.values.ravel()  
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(x_train, y_train_reshaped)
    return gb_model

def make_prediction(model: Sequential, X_test_scaled: np.ndarray) -> pd.DataFrame:
    predictions = model.predict(X_test_scaled)
    predictions_df = pd.DataFrame(predictions, columns=['Predicted_Time_taken(min)'])
    return predictions_df
