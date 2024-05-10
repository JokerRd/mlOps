from joblib import load
import pandas as pd


class IrisModel:

    def __init__(self):
        self.__init_model()

    def __init_model(self):
        self.model = load('model.joblib')

    def predict(self, data):
        return self.model.predict(data)
