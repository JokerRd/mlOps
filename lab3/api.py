from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from starlette import status
from starlette.responses import JSONResponse
from irismodel import IrisModel
import pandas as pd

app = FastAPI()
model = IrisModel()


class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/iris/predict", summary="Метод для предсказания класса для iris")
def predict_iris(iris_data: list[IrisData]):
    json_list = list(map(dict, iris_data))
    df = pd.DataFrame.from_records(json_list)
    result = model.predict(df.to_numpy())
    return list(result)
