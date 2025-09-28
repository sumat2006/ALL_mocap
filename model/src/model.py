import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, accuracy_score, classification_report,precision_score
from sklearn.metrics import confusion_matrix, classification_report
from scipy import signal
import os
from collections import Counter
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import numpy as np
from tqdm import tqdm
import math
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl

plt.rcParams['font.family'] = 'TH Sarabun New'  # Set default font for Thai language support
mpl.font_manager.fontManager.addfont('thsarabunnew-webfont.ttf')  # Ensure the font is loaded
mpl.rc('font', family='TH Sarabun New',size = 20)  # Set the font globally

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
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_classes)
        
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
        # print(x_norm)
        # ([print(i.dtype) for i in x_norm])
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
        
        # # Second Dense layer (output)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
    
    def get_config(self):
        """Return model configuration for saving"""
        return {
            'input_shape': self.input_shape,
            'n_classes': self.n_classes,
            'dropout': self.dropout_rate,
            'flatten_size': self.flatten_size
        }
        
        
        
class ImprovedCustomDataset(Dataset):
    def __init__(self, dataframe, chunk_size=121, label_encoder=None, is_test=False, 
                 balance_classes=False, min_sequence_length=50,zero_ratio=1,down_zero=True):
        """
        Improved dataset class for CNN time series classification
        
        Args:
            dataframe: Input dataframe
            chunk_size: Sequence length for time series (121 to match your diagram)
            label_encoder: Pre-fitted label encoder (for test data)
            is_test: Whether this is test data
            balance_classes: Whether to balance classes
            min_sequence_length: Minimum sequence length to keep
        """
        self.chunk_size = chunk_size
        self.min_sequence_length = min_sequence_length
        
        # Clean data
        self.data = dataframe.copy()
        self.data = self.data[~self.data.Label.isin(["cooldown", "error_redo", "break_time"])]
        
        # Handle label encoding
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.data["Label"] = self.label_encoder.fit_transform(self.data["Label"])
        else:
            self.label_encoder = label_encoder
            # Handle unseen labels in test data
            known_labels = set(label_encoder.classes_)
            mask = self.data["Label"].isin(known_labels)
            if not mask.all():
                print(f"Warning: Removing {(~mask).sum()} samples with unknown labels")
                self.data = self.data[mask]
            self.data["Label"] = label_encoder.transform(self.data["Label"])
        
        self.n_classes = len(self.label_encoder.classes_)
        
        # Remove timestamp if present
        if "timestamp_ms" in self.data.columns:
            self.data = self.data.drop(columns=["timestamp_ms"])
        
        # Convert to sequences
        self.sequences, self.labels = self._create_sequences()
        
        # Balance classes if requested and not test data
        if balance_classes and not is_test:
            self.sequences, self.labels = self._balance_classes()
        
        if down_zero and not is_test:
            self.sequences, self.labels = self._down_zero(zero_ratio = zero_ratio)
            
        
        print(f"Dataset created: {len(self.sequences)} sequences of shape {self.sequences.shape}")
        print(f"Classes: {self.n_classes}, Distribution: {Counter(self.labels.numpy())}")
    
    def _create_sequences(self):
        """Create sequences from grouped data"""
        # Group by consecutive labels
        self.data['group_id'] = (self.data['Label'] != self.data['Label'].shift()).cumsum()
        grouped_data = [group.drop('group_id', axis=1) for _, group in self.data.groupby('group_id')]
        
        print(f"Found {len(grouped_data)} label groups")
        
        # Filter by minimum length
        valid_groups = [group for group in grouped_data if len(group) >= self.min_sequence_length]
        print(f"Kept {len(valid_groups)} groups after length filtering")
        
        sequences = []
        labels = []
        
        for group in tqdm(valid_groups, desc="Processing sequences"):
            # Separate features and labels
            features = group.drop('Label', axis=1).values
            group_labels = group['Label'].values
            
            # Get the most common label in the sequence
            most_common_label = Counter(group_labels).most_common(1)[0][0]
            
            # Create fixed-length sequence
            if len(features) >= self.chunk_size:
                # If longer than chunk_size, use uniform sampling
                indices = np.linspace(0, len(features)-1, self.chunk_size, dtype=int)
                sequence = features[indices]
            else:
                # If shorter, pad with zeros at the end
                sequence = np.zeros((self.chunk_size, features.shape[1]))
                sequence[:len(features)] = features
            
            sequences.append(sequence)
            labels.append(most_common_label)
        
        return torch.FloatTensor(sequences), torch.LongTensor(labels)
    
    def _balance_classes(self):
        """Balance classes by undersampling majority classes"""
        # Get class counts
        unique_labels, counts = torch.unique(self.labels, return_counts=True)

        min_count = counts.min().item()
        
        print(f"Balancing classes to {min_count} samples each")
        
        balanced_sequences = []
        balanced_labels = []
        
        for label,counts in zip(unique_labels,counts):
            # Get indices for this class
            class_indices = (self.labels == label).nonzero(as_tuple=True)[0]
            # Sample min_count indices
            # if len(class_indices) > min_count and label.item() == 0:
            #     sampled_indices = class_indices[torch.randperm(len(class_indices))[:int(counts.item() * zero_ratio )]]
            if len(class_indices) > min_count:
                sampled_indices = class_indices[torch.randperm(len(class_indices))[:min_count]]
            else:
                sampled_indices = class_indices
            
            balanced_sequences.append(self.sequences[sampled_indices])
            balanced_labels.append(self.labels[sampled_indices])
        
        return torch.cat(balanced_sequences), torch.cat(balanced_labels)
    
    def _down_zero(self,zero_ratio = 1):
        """Balance classes by undersampling majority classes"""
        # Get class counts
        unique_labels, counts = torch.unique(self.labels, return_counts=True)

        
        print(f"down zero with {zero_ratio} ratio")
        
        balanced_sequences = []
        balanced_labels = []
        
        for label,counts in zip(unique_labels,counts):
            # Get indices for this class
            class_indices = (self.labels == label).nonzero(as_tuple=True)[0]
            # Sample min_count indices
            if label.item() == 0:
                sampled_indices = class_indices[torch.randperm(len(class_indices))[:int(counts.item() * zero_ratio )]]
            else:
                sampled_indices = class_indices
            
            balanced_sequences.append(self.sequences[sampled_indices])
            balanced_labels.append(self.labels[sampled_indices])
        
        return torch.cat(balanced_sequences), torch.cat(balanced_labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
    
    def get_info(self):
        """Return dataset information"""
        return {
            'n_samples': len(self.sequences),
            'sequence_length': self.chunk_size,
            'n_features': self.sequences.shape[2],
            'n_classes': self.n_classes,
            'label_encoder': self.label_encoder,
            'class_names': self.label_encoder.classes_,
            'input_shape': (self.chunk_size, self.sequences.shape[2])
        }
        
class WarmupCosineScheduler:
    """Custom warmup + cosine annealing scheduler"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr
    
    
def train_model(model, train_loader, val_loader=None, num_epochs=100, learning_rate=0.001, 
                device='cuda', scheduler_type='warmup_cosine', patience=20, save_best=True,
                save_dir='models', model_name=None):
    """
    Training function for CNN time series classifier with model saving
    """
    model = model.to(device)
    
    # Loss function for multi-class classification
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Scheduler setup
    if scheduler_type == 'warmup_cosine':
        warmup_epochs = max(1, num_epochs // 10)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, num_epochs)
    elif scheduler_type == 'cosine_restart':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    elif scheduler_type == 'reduce_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=10, min_lr=1e-6, factor=0.5, verbose=True
        )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'learning_rates': []
    }
    
    # Early stopping
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    print(f"🚀 Starting CNN training on {device}")
    print(f"📊 Scheduler: {scheduler_type}, Epochs: {num_epochs}, LR: {learning_rate}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [TRAIN]')
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # print(batch_x.size())
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{train_loss/(train_pbar.n+1):.4f}',
                'Acc': f'{100*train_correct/train_total:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['learning_rates'].append(current_lr)
        
        # Validation phase
        val_loss, val_accuracy, val_f1 = 0.0, 0.0, 0.0
        
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            val_f1 = f1_score(all_labels, all_preds, average='weighted') * 100
            
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_accuracy)
            history['val_f1'].append(val_f1)
            
            # Early stopping and best model saving
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f"💾 New best model! Val Acc: {val_accuracy:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping after {patience} epochs without improvement")
                break
        
        # Step scheduler
        if scheduler_type == 'reduce_plateau' and val_loader is not None:
            scheduler.step(val_loss)
        elif scheduler_type in ['warmup_cosine', 'cosine_restart']:
            scheduler.step()
        
        # Print epoch summary
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}%')
        if val_loader is not None:
            print(f'Val   - Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%, F1: {val_f1:.2f}%')
        print(f'LR: {current_lr:.6f}, Patience: {patience_counter}/{patience}')
        print('-' * 70)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"🔄 Loaded best model with validation accuracy: {best_val_acc:.2f}%")
    
    return model, history



def evaluate_model(model, test_loader, label_encoder, device='cuda', save_dir='evaluation_results'):
    """Comprehensive model evaluation function with visualizations"""
    model.eval()
    model.to(device)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    print("🔍 Evaluating model...")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc='Evaluating'):
            print(batch_x.size())
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
    f1_weighted = f1_score(all_labels, all_preds, average='weighted') * 100
    recall = recall_score(all_labels, all_preds, average='weighted') * 100
    precision = precision_score(all_labels, all_preds, average='weighted') * 100
    avg_loss = total_loss / len(test_loader)
    
    # Print detailed results
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"{'='*50}")
    print(f"Test Loss:      {avg_loss:.4f}")
    print(f"Accuracy:       {accuracy:.2f}%")
    print(f"Precision:      {precision:.2f}%")
    print(f"F1 (Macro):     {f1_macro:.2f}%")
    print(f"F1 (Weighted):  {f1_weighted:.2f}%")
    print(f"Recall:         {recall:.2f}%")
    print(f"{'='*50}")
    
    # Create visualizations (images only)
    create_confusion_matrix(all_labels, all_preds, label_encoder.classes_, save_dir)
    create_per_class_table(all_labels, all_preds, label_encoder.classes_, save_dir)
    create_classification_report_table(all_labels, all_preds, label_encoder.classes_, save_dir)
    
    # Per-class breakdown (console output)
    print(f"\n📋 PER-CLASS BREAKDOWN:")
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = (all_labels == i)
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == i).mean() * 100
            print(f"{class_name:15}: {class_acc:.1f}% ({class_mask.sum()} samples)")
    
    # Detailed classification report (console output)
    print(f"\n📈 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
    
    print(f"\n💾 Images saved to: {save_dir}/")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'recall': recall,
        'precision': precision,
        'loss': avg_loss,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs
    }

def create_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """Create and save confusion matrix heatmap (images only)"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_per_class_table(y_true, y_pred, class_names, save_dir):
    """Create per-class breakdown table image"""
    per_class_data = []
    
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        if class_mask.sum() > 0:
            true_positives = ((y_pred == i) & (y_true == i)).sum()
            total_actual = class_mask.sum()
            total_predicted = (y_pred == i).sum()
            
            accuracy = (true_positives / total_actual) * 100 if total_actual > 0 else 0
            precision = (true_positives / total_predicted) * 100 if total_predicted > 0 else 0
            recall = accuracy  # Same as accuracy for per-class
            
            per_class_data.append({
                'Class': class_name,
                'Accuracy (%)': f"{accuracy:.1f}",
                'Precision (%)': f"{precision:.1f}",
                'Recall (%)': f"{recall:.1f}",
                'Support': int(total_actual)
            })
    
    df = pd.DataFrame(per_class_data)
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(12, max(6, len(class_names) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                    cellLoc='center', loc='center',
                    colColours=['lightblue'] * len(df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(df.columns)):
        table[(0, i)].set_text_props(weight='bold')
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(color='white')
    
    plt.title('Per-Class Performance Breakdown', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'{save_dir}/per_class_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_classification_report_table(y_true, y_pred, class_names, save_dir):
    """Create detailed classification report table image"""
    # Get classification report as dictionary
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                 output_dict=True, zero_division=0)
    
    # Create DataFrame for individual classes
    class_data = []
    for class_name in class_names:
        if class_name in report:
            class_data.append({
                'Class': class_name,
                'Precision': f"{report[class_name]['precision']:.3f}",
                'Recall': f"{report[class_name]['recall']:.3f}",
                'F1-Score': f"{report[class_name]['f1-score']:.3f}",
                'Support': int(report[class_name]['support'])
            })
    
    df_classes = pd.DataFrame(class_data)
    
    # Create summary data
    summary_data = [
        ['Macro Avg', f"{report['macro avg']['precision']:.3f}", 
         f"{report['macro avg']['recall']:.3f}", f"{report['macro avg']['f1-score']:.3f}", 
         int(report['macro avg']['support'])],
        ['Weighted Avg', f"{report['weighted avg']['precision']:.3f}", 
         f"{report['weighted avg']['recall']:.3f}", f"{report['weighted avg']['f1-score']:.3f}", 
         int(report['weighted avg']['support'])],
        ['Accuracy', '', '', f"{report['accuracy']:.3f}", 
         int(report['macro avg']['support'])]
    ]
    
    # Combine class data with summary
    all_data = df_classes.values.tolist() + [['', '', '', '', '']] + summary_data
    all_columns = df_classes.columns.tolist()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, max(8, len(all_data) * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=all_data, colLabels=all_columns, 
                    cellLoc='center', loc='center',
                    colColours=['lightblue'] * len(all_columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style the header
    for i in range(len(all_columns)):
        table[(0, i)].set_text_props(weight='bold')
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(color='white')
    
    # Style summary rows
    summary_start_idx = len(df_classes) + 2  # +1 for header, +1 for empty row
    for i in range(summary_start_idx, len(all_data) + 1):
        for j in range(len(all_columns)):
            table[(i, j)].set_facecolor('#f0f0f0')
            if i > summary_start_idx:  # Don't bold the empty row
                table[(i, j)].set_text_props(weight='bold')
    
    plt.title('Classification Report', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'{save_dir}/classification_report.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def save_model(model, label_encoder, dataset_info, history, scheduler_type, 
               save_dir='models', model_name=None):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Generate model name if not provided
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"cnn_timeseries_model_{timestamp}"
    
    # Ensure .pth extension
    if not model_name.endswith('.pth'):
        model_name += '.pth'

    full_model_path = os.path.join(save_dir, "Full_model_"+model_name)
    model_path = os.path.join(save_dir, model_name)

    # Get model configuration
    model_config = model.get_config()
    
    # Prepare save dictionary
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'label_encoder': label_encoder,
        'dataset_info': {
            'n_samples': dataset_info['n_samples'],
            'sequence_length': dataset_info['sequence_length'],
            'n_features': dataset_info['n_features'],
            'n_classes': dataset_info['n_classes'],
            'class_names': list(dataset_info['class_names']),
            'input_shape': dataset_info['input_shape']
        },
        'training_info': {
            'scheduler_used': scheduler_type,
            'final_train_acc': history['train_acc'][-1] if history['train_acc'] else None,
            'final_val_acc': history['val_acc'][-1] if 'val_acc' in history and history['val_acc'] else None,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'final_val_loss': history['val_loss'][-1] if 'val_loss' in history and history['val_loss'] else None,
            'total_epochs': len(history['train_loss']) if history['train_loss'] else 0
        },
        'training_history': history,
        'save_timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__
    }
    
    # Save the model
    try:
        torch.save(model, full_model_path)
        torch.save(save_dict, model_path)
        print(f"✅ Model saved successfully to: {model_path}")
        
        # Save training history as JSON for easy inspection
        history_path = model_path.replace('.pth', '_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_history = {}
            for key, value in history.items():
                if isinstance(value, list):
                    json_history[key] = value
                else:
                    json_history[key] = str(value)
            json.dump(json_history, f, indent=2)
        print(f"📊 Training history saved to: {history_path}")
        
        # Print model summary
        print(f"\n📋 MODEL SAVE SUMMARY:")
        print(f"{'='*50}")
        print(f"Model Name:     {model_name}")
        print(f"Input Shape:    {model_config['input_shape']}")
        print(f"Classes:        {model_config['n_classes']}")
        print(f"Parameters:     {sum(p.numel() for p in model.parameters()):,}")
        print(f"Scheduler:      {scheduler_type}")
        print(f"Save Time:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")
        
        return model_path
        
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")
        return None
    
def load_model(model_path, device='cuda'):
    """
    Load a saved CNN time series classifier model
    
    Args:
        model_path: Path to the saved .pth model file
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        tuple: (model, model_info) where model_info contains metadata
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Load the saved data
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration
    model_config = checkpoint['model_config']
    input_shape = model_config['input_shape']
    n_classes = model_config['n_classes']
    dropout = model_config['dropout']
    
    # Create model with the same configuration
    model = CNNTimeSeriesClassifier(
        input_shape=input_shape,
        n_classes=n_classes,
        dropout=dropout
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Extract other useful information
    model_info = {
        'label_encoder': checkpoint['label_encoder'],
        'dataset_info': checkpoint['dataset_info'],
        'training_info': checkpoint['training_info'],
        'class_names': checkpoint['dataset_info']['class_names'],
        'input_shape': input_shape,
        'n_classes': n_classes,
        'save_timestamp': checkpoint.get('save_timestamp', 'Unknown'),
        'pytorch_version': checkpoint.get('pytorch_version', 'Unknown')
    }
    
    print(f"Model loaded successfully!")
    print(f"Input Shape: {input_shape}")
    print(f"Number of Classes: {n_classes}")
    print(f"Class Names: {model_info['class_names']}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, model_info

def convert_data(features):
    chunk_size = 50
    
    if len(features) >= 50:
        # If longer than chunk_size, use uniform sampling
        indices = np.linspace(0, len(features)-1, chunk_size, dtype=int)
        sequence = features[indices]
    else:
        # If shorter, pad with zeros at the end
        sequence = np.zeros((chunk_size, features.shape[1]))
        sequence[:len(features)] = features

    
    if torch.cuda.is_available():
        return torch.tensor(sequence).to("cuda")
    else:
        return torch.tensor(sequence)
    
    