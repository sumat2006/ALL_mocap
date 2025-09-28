import imp
from model import CNNTimeSeriesClassifier,ImprovedCustomDataset,train_model,save_model,evaluate_model
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import json
import numpy as np
CONFIG = {
        'model_name':"test",
        'save_dir': 'saved_models',
        'train_data_path': r'C:\SKB_co_Project\Mocap\git\Sign_Language_Detection_local\meta_data\normal_train_data.csv',
        # 'test_data_path': r'C:\SKB_co_Project\Mocap\git\Sign_Language_Detection_local\meta_data\indicator-20250727T060250Z-1-001\indicator\data\20250715_111750_DATA_INDICATOR_sensor.csv',
        'test_data_path': r'C:\SKB_co_Project\Mocap\git\Sign_Language_Detection_local\meta_data\eval_data.csv',
        'chunk_size': 30,
        'batch_size': 64,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'dropout_rate': 0.3,
        'scheduler_type': 'warmup_cosine',  # Options: 'warmup_cosine', 'cosine_restart', 'reduce_plateau'
        'balance_classes': False,
        'patience': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
print(f"Using device: {CONFIG['device']}")

train_dataset = ImprovedCustomDataset(
    dataframe = pd.read_csv(CONFIG['train_data_path']),
    chunk_size = CONFIG['chunk_size'],
    balance_classes = CONFIG['balance_classes'],
)

# Create data loaders with train/validation split
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_subset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_subset, batch_size=CONFIG['batch_size'], shuffle=False)

 # Get dataset info
dataset_info = train_dataset.get_info()
input_shape = dataset_info['input_shape']
n_classes = dataset_info['n_classes']

print(f"📊 Dataset Info:")
print(f"   Input Shape: {input_shape}")
print(f"   Classes: {n_classes}")
print(f"   Class Names: {list(dataset_info['class_names'])}")
print(f"📈 Data Split: Train={len(train_subset)}, Val={len(val_subset)}")

cls_model = CNNTimeSeriesClassifier(
                                    input_shape=input_shape,
                                    n_classes=n_classes,
                                    dropout=CONFIG['dropout_rate']
                                    )

print(f"🏗️  Model created with {sum(p.numel() for p in cls_model.parameters())} parameters")


trained_model, history = train_model(
        model=cls_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        device=CONFIG['device'],
        scheduler_type=CONFIG['scheduler_type'],
        patience=CONFIG['patience']
    )



timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"{CONFIG['model_name']}_{timestamp}"

saved_model_path = save_model(
        model=trained_model,
        label_encoder=dataset_info['label_encoder'],
        dataset_info=dataset_info,
        history=history,
        scheduler_type=CONFIG['scheduler_type'],
        save_dir=CONFIG['save_dir'],
        model_name=model_name
    )

# Load test data if available
try:
    test_df = pd.read_csv(CONFIG['test_data_path'])
    
    test_dataset = ImprovedCustomDataset(
        dataframe=test_df,
        chunk_size=CONFIG['chunk_size'],
        label_encoder=dataset_info['label_encoder'],
        is_test=True,
        balance_classes=False
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"✅ Test data loaded: {len(test_dataset)} samples")
    
    # Evaluate on test set
    eval_results = evaluate_model(
        trained_model, test_loader, 
        dataset_info['label_encoder'], CONFIG['device']
    )
    
    # Save evaluation results
    if saved_model_path:
        eval_path = saved_model_path.replace('.pth', '_evaluation.json')
        with open(eval_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_eval = {}
            for key, value in eval_results.items():
                if isinstance(value, np.ndarray):
                    json_eval[key] = value.tolist()
                else:
                    json_eval[key] = value
            json.dump(json_eval, f, indent=2)
        print(f"📊 Evaluation results saved to: {eval_path}")
    
except FileNotFoundError:
    print("⚠️  No test data found. Skipping test evaluation.")
    eval_results = None