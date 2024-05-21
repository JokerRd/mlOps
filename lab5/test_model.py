from joblib import load
import pandas as pd
from sklearn.metrics import r2_score


def test_dataset_without_noise():
    reg_model = load("model.joblib")
    X_test = pd.read_csv("X_test.csv", header=0)
    y_test = pd.read_csv("y_test.csv")
    y_test_predict = reg_model.predict(X_test)
    r2 = r2_score(y_test, y_test_predict)
    assert r2 > 0.85


def test_dataset_with_noise():
    reg_model = load("model.joblib")
    X_noise = pd.read_csv("X_noise.csv", header=0)
    y_noise = pd.read_csv("y_noise.csv")
    y_noise_predict = reg_model.predict(X_noise)
    r2 = r2_score(y_noise, y_noise_predict)
    assert r2 > 0.85
