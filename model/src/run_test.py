import torch 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import json
from model import CNNTimeSeriesClassifier,ImprovedCustomDataset,convert_data
import os
import yaml
from pathlib import Path

with open("config.yaml", 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

with open(config["rollback"],'r',encoding="utf-8") as f:
    rollback = json.load(f)
    
model_path = config["model"]
classifier =  config["classifier"]


if torch.cuda.is_available():
    model = torch.load(f"{model_path}",weights_only=False)
    classifier = torch.load(f"{classifier}",weights_only=False)
    model.to("cuda")
else:
    model = torch.load(f"{model_path}",weights_only=False,map_location=torch.device('cpu'))
    classifier = torch.load(f"{classifier}",weights_only=False,map_location=torch.device('cpu'))
    
model.double()
model.eval()
test_df = pd.read_csv(config["test_df"])
test_df = test_df[~(test_df.Label.isin(["error_redo","break_time"]))].reset_index(drop=True)
test = test_df.drop(columns=["Label","timestamp_ms"]).values
Label = test_df["Label"].values
data = []
pv_label = ""
y_true = []
y_pred = []
for test_data,lab in zip(test,Label):
    # print(lab)
    if lab!=pv_label and pv_label!="":
        # print(f"class from {pv_label} --> {lab}")
        if len(data) < 30:
            data = []
            continue
        data = torch.tensor(data)
        print("convert_data")
        tas = convert_data(data)
        # print(tas)
        answer = torch.argmax(model(tas.unsqueeze(0)))
        finalans = rollback[str(answer.item())]
        print(f"the current class is {finalans} of class {pv_label}")
        data = []
    else:
        data.append(test_data)
        
    pv_label = lab

