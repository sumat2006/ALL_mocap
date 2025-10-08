import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import deque
from transformers import pipeline,WhisperForConditionalGeneration
from peft import PeftModel


thes = 5

predictions = []
data = []
ft = []
sta = []
state = False
a = []
DELTA_ACC_THRESHOLD = 0
DELTA_GYRO_THRESHOLD = 0
DELTA_FLEX_THRESHOLD = 0
WINDOW_SIZE = 2   # ใช้ย้อนหลัง 5 segment
ini_state = False

class S2S:
    def __init__(self,model_name):
        
        self.model = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")
        
        
    def predict(self,text,persona=None):
        outputs = self.model(text, max_new_tokens=25, do_sample=True, temperature=0.9, top_k=50, top_p=0.95, no_repeat_ngram_size=2,typical_p=1.)
        return outputs
    
    
class Whisper:
    def __init__(self,model_name,peft_model_id):
        base_model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model = PeftModel.from_pretrained(base_model, peft_model_id)
        self.model = pipeline(task="automatic-speech-recognition", model=model, torch_dtype=torch.bfloat16, device_map="auto")
        
        
    def predict(self,speech):
        outputs = self.model("audio.mp3", generate_kwargs={"language":"<|th|>", "task":"transcribe"}, batch_size=16)["text"]
        return outputs
    
class Thonburain:
    def __init__(self,model_name):
        self.model = pipeline(task="automatic-speech-recognition", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")
        
        
    def predict(self,speech,sr):
        outputs = self.model({"array": speech, "sampling_rate": sr}, generate_kwargs={"language":"<|th|>", "task":"transcribe"}, batch_size=16)["text"]
        return outputs

class RapidChangeWindowClassifier:
    
    def __init__(self, acc_th=DELTA_ACC_THRESHOLD, gyro_th=DELTA_GYRO_THRESHOLD, flex_th=DELTA_FLEX_THRESHOLD, window_size=WINDOW_SIZE,use_gyro=True):
        self.acc_th = acc_th
        self.gyro_th = gyro_th
        self.flex_th = flex_th
        self.window_size = window_size
        self.use_gyro = use_gyro
        # deque เก็บค่าล่าสุด (ทำงานเหมือน sliding window)
        self.acc_hist = deque(maxlen=window_size)
        self.gyro_hist = deque(maxlen=window_size)
        self.flex_hist = deque(maxlen=window_size)

    def compute_features(self, row):
        acc_mag = np.sqrt(row[1]**2 + row[2]**2 + row[3]**2) + np.sqrt(row[15]**2 + row[16]**2 + row[17]**2)
        gyro_mag = np.sqrt(row[3]**2 + row[4]**2 + row[5]**2) + np.sqrt(row[18]**2 + row[19]**2 + row[20]**2)
        flex_mean = np.mean([row[10], row[11], row[12], row[13], row[14],row[24], row[25], row[26], row[27], row[28]])
        # print(acc_mag,gyro_mag,flex_mean)
        return acc_mag, gyro_mag, flex_mean

    def classify(self, row):
        acc_mag, gyro_mag, flex_mean = self.compute_features(row)


        self.acc_hist.append(acc_mag)
        self.gyro_hist.append(gyro_mag)
        self.flex_hist.append(flex_mean)


        delta_acc = max(self.acc_hist) - min(self.acc_hist)
        delta_gyro = max(self.gyro_hist) - min(self.gyro_hist)
        delta_flex = max(self.flex_hist) - min(self.flex_hist)
        

        # 
        if self.use_gyro:
            if (delta_gyro > self.gyro_th):
            # if seq >= 2:
                return True
            else:
                return False
        else:
            seq = sum([(delta_acc > self.acc_th), (delta_gyro > self.gyro_th), (delta_flex > self.flex_th)])
            if seq >= 2:
                return True
            else:
                return False
    def padding(self,data,thes):
        answer = []
        features = np.array(data[:-thes])
        if len(features) >= 50:
            # If longer than chunk_size, use uniform sampling
            indices = np.linspace(0, len(features)-1, 50, dtype=int)
            sequence = features[indices]
        else:
            # If shorter, pad with zeros at the end
            sequence = np.zeros((50, features.shape[1]))
            sequence[:len(features)] = features
        return sequence
            
    def pred(self,rows,predictions,data,ft,state,sta):
        for row in rows:
            # print(row)
            st = self.classify(row)
            
            a.append(st)
            if st :### if predict true 
                state = True
                ### add data re false array and add true array
                predictions.append(st)
                data.append(row)
                ft = []
                sta.append(st)
            elif not st and state: ### if predict false and previous true
                ### add data and add false array
                
                ft.append(st)
                predictions.append(st)
                data.append(row)
            
        return ft,predictions,data,sta,state
            
class CNNTimeSeriesClassifier(nn.Module):
    def __init__(self, input_shape, n_classes, dropout=0.3):
        """
        CNN-based Time Series Classifier following your architecture diagram
        
        Args:
            input_shape: Tuple (sequence_length, n_features) - e.g., (121, 21)
            n_classes: Number of output classes
            dropout: Dropout rate
        """
        super(CNNTimeSeriesClassifier, self).__init__()
        
        seq_len, n_features = input_shape
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.dropout_rate = dropout
        
        # Input normalization layer
        self.normalization = nn.BatchNorm1d(n_features)
        
        # Reshape for 2D convolution: (batch, channels, height, width)
        # We'll treat sequence as height and features as width, with 1 channel
        # Input shape: (None, 1, seq_len, n_features) - e.g., (None, 1, 121, 21)
        # First Conv2D block
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=32, 
            kernel_size=(3, 3), 
            padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # Calculate shape after first conv+pool
        # After conv1: (None, 32, 121, 21) -> (None, 32, 119, 19) with padding=1
        # After pool1: (None, 32, 119, 19) -> (None, 32, 59, 9)
        h1, w1 = seq_len // 2, n_features // 2
        
        # Second Conv2D block
        self.conv2 = nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=(3, 3), 
            padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # Calculate shape after second conv+pool
        # After conv2: (None, 64, 59, 9) -> (None, 64, 57, 7) with padding=1
        # After pool2: (None, 64, 57, 7) -> (None, 64, 28, 3)
        h2, w2 = h1 // 2, w1 // 2
        
        # Third Conv2D block
        self.conv3 = nn.Conv2d(
            in_channels=64, 
            out_channels=64, 
            kernel_size=(3, 3), 
            padding=1
        )
        
        # Calculate final conv output shape
        # After conv3: (None, 64, 28, 3) -> (None, 64, 26, 1) with padding=1
        h3, w3 = h2, w2
        
        # Calculate flattened size dynamically
        self.flatten_size = 64 * h3 * w3
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc2 = nn.Linear(64, n_classes)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Input shape: (batch_size, seq_len, n_features)
        batch_size = x.size(0)
        
        # Normalize along feature dimension
        # Reshape for BatchNorm1d: (batch_size * seq_len, n_features)
        x_norm = x.view(-1, x.size(2))
        x_norm = self.normalization(x_norm)
        x = x_norm.view(batch_size, x.size(1), x.size(2))
        
        # Reshape for 2D convolution: (batch_size, 1, seq_len, n_features)
        x = x.unsqueeze(1)
        
        # First Conv2D + MaxPool2D
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Second Conv2D + MaxPool2D
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Third Conv2D
        x = self.conv3(x)
        x = self.relu(x)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # First Dense layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second Dense layer (output)
        x = self.fc2(x)
        
        return x
    
    def get_config(self):
        """Return model configuration for saving"""
        return {
            'input_shape': self.input_shape,
            'n_classes': self.n_classes,
            'dropout': self.dropout_rate,
            'flatten_size': self.flatten_size
        }
        
        
