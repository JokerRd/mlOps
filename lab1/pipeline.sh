#!/bin/bash
pip install ucimlrepo
pip install scikit-learn
pip install pandas
python data_creation.py && python model_preprocessing.py && python model_preparation.py && python model_testing.py