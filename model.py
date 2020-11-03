import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OutputCodeClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
import warnings
from data_loader.data import *
from typing import List, Dict, Optional
import pickle

warnings.filterwarnings("ignore")

dataset = pd.read_csv("dataset.csv")

data_copy = dataset.copy()

# generate synthetic data

data = data_copy.sample(n=5000, replace=True)

# data.to_csv("synthetic_data.csv")


# Function to load data

# def load_data(file_path : str,col_to_drop : List[str],target_col :str):
#     data = pd.read_csv(file_path)
#     targets = data[target_col]
#     targets = np.array(targets)
#     targets = targets.reshape(-1,1)
#     inputs = inputs = data.drop(col_to_drop,axis=1)
#     return inputs, targets


# Load Data using the data loader module

inputs, target = load_data("synthetic_data.csv", ["id",
                           "filename", "label"], "label")

print(inputs.shape)

# print(inp_1)

# seperate target and inputs

# target = data["label"]

# inputs = data.drop(["filename","label"],axis=1)

# print(type(inputs))


# Create one hot encoder object

ohe = OneHotEncoder()

Or = OrdinalEncoder()

# print(type(target))

target = np.array(target)

target = target.reshape(-1, 1)

targets = Or.fit_transform(target)

print(type(targets))

# inv = ohe.inverse_transform(targets)

# print(inv)


input_train, input_test, target_train, target_test = train_test_split(
    inputs, targets, test_size=0.3)

ecc = OutputCodeClassifier(KNeighborsClassifier(), code_size=2)

ecc.fit(input_train, target_train)

pred = ecc.predict(input_test)

print(accuracy_score(target_test, pred))

print(f1_score(target_test, pred, average="weighted"))

print("="*30)
print(Or.inverse_transform(np.array(pred).reshape(-1, 1)))

print("--"*20)

print(Or.inverse_transform(target_test))
#
# with open("Saved_model.sav","wb") as f:
#     pickle.dump(ecc,f)

pickle.dump(ecc, open("Saved_model.sav", "wb"))

# with open("Encodings.sav","wb") as f:
#     pickle.dump(Or,f)

pickle.dump(Or, open("Encodings.sav", "wb"))



