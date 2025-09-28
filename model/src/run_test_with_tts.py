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
import pygame
import numpy as np
import torch
import pygame
import os
from pathlib import Path
from collections import deque # Optimized for adding/removing from both ends
import time
import pandas as pd
import random

def split_label_segments(df):
    if 'Label' not in df.columns or df.empty:
        return []
    
    list_label_stamp = []
    i = 0
    i_start = i
    start_label = df['Label'].iloc[i_start]

    while i < len(df['Label']):
        if df['Label'].iloc[i] == start_label:
            i += 1
        else:
            list_label_stamp.append((start_label, i_start, i - 1))
            start_label = df['Label'].iloc[i]
            i_start = i
            i += 1
            
    if i_start < len(df['Label']):
        list_label_stamp.append((start_label, i_start, len(df['Label']) - 1))

    return list_label_stamp


def shuffle_dataframe_by_segments(df):
    label_segments = split_label_segments(df)
    random.shuffle(label_segments)
    shuffled_df_list = []
    for segment in label_segments:
        start_index = segment[1]
        end_index = segment[2]
        shuffled_df_list.append(df.iloc[start_index:end_index + 1])
    shuffled_df = pd.concat(shuffled_df_list, ignore_index=True)
    
    return shuffled_df


script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = Path(script_dir) / "config.yaml"
print(config_path)

with open(config_path, 'r',encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print(config)


with open(Path(script_dir) / config["rollback"],'r',encoding="utf-8") as f:
    rollback = json.load(f)

model_path = Path(script_dir) / config["model"]
classifier = Path(script_dir) / config["classifier"]


if torch.cuda.is_available():
    model = torch.load(f"{model_path}",weights_only=False)
    classifier = torch.load(f"{classifier}",weights_only=False)
    model.to("cuda")
else:
    model = torch.load(f"{model_path}",weights_only=False,map_location=torch.device('cpu'))
    classifier = torch.load(f"{classifier}",weights_only=False,map_location=torch.device('cpu'))
    
model.double()
model.eval()

test_df = pd.read_csv(Path(script_dir) / config["test_df"])
test_df = shuffle_dataframe_by_segments(test_df)

test_df = test_df[~(test_df.Label.isin(["error_redo","break_time"]))].reset_index(drop=True)
test = test_df.drop(columns=["Label","timestamp_ms"]).values
Label = test_df["Label"].values


WINDOW_SIZE = 30 
CONFIDENCE_THRESHOLD = 0.5
STABILITY_THRESHOLD = 5

data_window = deque(maxlen=WINDOW_SIZE)
prediction_buffer = deque(maxlen=STABILITY_THRESHOLD)
last_played_sound = ""

pygame.mixer.init()

print("--- Starting Real-Time Streaming Simulation ---")
for i, test_data in enumerate(test):
    
    data_window.append(test_data)
    if len(data_window) == WINDOW_SIZE:
        numpy_data = np.array(data_window, dtype=np.double)
        data_tensor = torch.from_numpy(numpy_data)
        tas = convert_data(data_tensor)
        
        with torch.no_grad():
            raw_output = model(tas.unsqueeze(0))
            probabilities = torch.nn.functional.softmax(raw_output, dim=1)
            max_probability, answer_index = torch.max(probabilities, 1)

        confidence = max_probability.item()
        predicted_class = rollback[str(answer_index.item())]
        
        if confidence >= CONFIDENCE_THRESHOLD:
            prediction_buffer.append(predicted_class)
        else:
            prediction_buffer.append("nothing")

        is_stable = len(prediction_buffer) == STABILITY_THRESHOLD and len(set(prediction_buffer)) == 1
        
        stable_prediction = prediction_buffer[0] if is_stable else "nothing"
        if stable_prediction != "nothing" and stable_prediction != last_played_sound and not pygame.mixer.music.get_busy():
            print(f"Stable prediction at frame {i}: '{stable_prediction}' with {confidence:.2%} confidence. true Label {Label[i]}")
            sound_file_path = Path(script_dir) / f"asset/sound/data_0/output_{stable_prediction}.wav"
            if os.path.exists(sound_file_path):
                pygame.mixer.music.load(sound_file_path)
                pygame.mixer.music.play()
                last_played_sound = stable_prediction
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                time.sleep(0.5)
            else:
                print(f"Warning: Sound file not found for class '{stable_prediction}'")
        
        elif stable_prediction == "nothing":
            last_played_sound = "nothing"
    