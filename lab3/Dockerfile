FROM python:3.11.4

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN \
    --mount=type=cache,target=/var/cache/apt \
    pip install --no-cache-dir --upgrade -r /code/requirements.txt
ENV PYTHONPATH /code
COPY ./irismodel.py /code/irismodel.py
COPY ./api.py /code/api.py
COPY ./__init__.py /code/__init__.py
COPY ./model.joblib /code/model.joblib
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]