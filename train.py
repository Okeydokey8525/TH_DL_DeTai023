import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt

# Import Dataset and Models from current directory
from data_preprocess import MultimodalECGDataset
from models import MultimodalAttentionNetwork

def calculate_metrics(y_true, y_pred):
    y_pred_bin = (y_pred > 0.5).astype(np.float32)
    acc = np.mean(y_true == y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        auc = 0.5
    return acc, f1, auc

def train_multimodal_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=5):
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'train_auc': [], 'val_auc': [],
        'mean_attn_seq': [], 'mean_attn_img': [], 'mean_attn_tab': []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_train_true = []
        all_train_pred = []
        
        for signals, images, tabulars, labels in train_loader:
            signals = signals.to(device)
            images = images.to(device)
            tabulars = tabulars.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs, attn_w = model(signals, images, tabulars)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * signals.size(0)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_train_true.append(labels.cpu().numpy())
            all_train_pred.append(probs)
            
        if scheduler:
            scheduler.step()
            
        train_loss /= len(train_loader.dataset)
        all_train_true = np.concatenate(all_train_true, axis=0)
        all_train_pred = np.concatenate(all_train_pred, axis=0)
        train_acc, train_f1, train_auc = calculate_metrics(all_train_true, all_train_pred)
        
        model.eval()
        val_loss = 0.0
        all_val_true = []
        all_val_pred = []
        all_attn_weights = []
        
        with torch.no_grad():
            for signals, images, tabulars, labels in val_loader:
                signals = signals.to(device)
                images = images.to(device)
                tabulars = tabulars.to(device)
                labels = labels.to(device)
                
                outputs, attn_w = model(signals, images, tabulars)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * signals.size(0)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_val_true.append(labels.cpu().numpy())
                all_val_pred.append(probs)
                
                all_attn_weights.append(attn_w.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        all_val_true = np.concatenate(all_val_true, axis=0)
        all_val_pred = np.concatenate(all_val_pred, axis=0)
        all_attn_weights = np.concatenate(all_attn_weights, axis=0)
        
        mean_attn = np.mean(all_attn_weights, axis=0)
        val_acc, val_f1, val_auc = calculate_metrics(all_val_true, all_val_pred)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f} - AUC: {val_auc:.4f} | "
              f"Attn (Seq/Img/Tab): {mean_attn[0]:.3f}/{mean_attn[1]:.3f}/{mean_attn[2]:.3f}")
              
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['mean_attn_seq'].append(float(mean_attn[0]))
        history['mean_attn_img'].append(float(mean_attn[1]))
        history['mean_attn_tab'].append(float(mean_attn[2]))
        
    return history

def evaluate_multimodal_test(model, test_loader, device):
    model.eval()
    all_true = []
    all_pred = []
    all_attn_weights = []
    
    with torch.no_grad():
        for signals, images, tabulars, labels in test_loader:
            signals = signals.to(device)
            images = images.to(device)
            tabulars = tabulars.to(device)
            
            outputs, attn_w = model(signals, images, tabulars)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_true.append(labels.numpy())
            all_pred.append(probs)
            all_attn_weights.append(attn_w.cpu().numpy())
            
    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    all_attn_weights = np.concatenate(all_attn_weights, axis=0)
    
    acc, f1, auc = calculate_metrics(all_true, all_pred)
    mean_attn = np.mean(all_attn_weights, axis=0)
    return acc, f1, auc, mean_attn

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_file = 'processed/ptbxl_processed_bai23.npz'
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Processed data file {data_file} not found. Run data_preprocess.py first.")
        
    print("Loading preprocessed dataset...")
    data = np.load(data_file)
    
    train_dataset = MultimodalECGDataset(data['x_train_seq'], data['x_train_img'], data['x_train_num'], data['y_train'])
    val_dataset = MultimodalECGDataset(data['x_val_seq'], data['x_val_img'], data['x_val_num'], data['y_val'])
    test_dataset = MultimodalECGDataset(data['x_test_seq'], data['x_test_img'], data['x_test_num'], data['y_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    epochs = 8
    
    print("\n--- TRAINING MULTIMODAL MODEL (Bai 23) ---")
    model = MultimodalAttentionNetwork().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    multimodal_history = train_multimodal_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler=scheduler,
        device=device, epochs=epochs
    )
    
    test_acc, test_f1, test_auc, test_mean_attn = evaluate_multimodal_test(model, test_loader, device)
    print(f"\n[MULTIMODAL TEST RESULTS] Accuracy: {test_acc:.4f} | F1 Macro: {test_f1:.4f} | AUC Macro: {test_auc:.4f}")
    print(f"[FINAL ATTENTION WEIGHTS] Sequence: {test_mean_attn[0]:.4f} | Image: {test_mean_attn[1]:.4f} | Tabular: {test_mean_attn[2]:.4f}")
    
    torch.save(model.state_dict(), 'processed/multimodal_ecg.pt')
    
    history_data = {
        'multimodal': multimodal_history,
        'results': {
            'acc': test_acc, 'f1': test_f1, 'auc': test_auc,
            'mean_attention': test_mean_attn.tolist()
        }
    }
    with open('processed/history_bai23.json', 'w') as f:
        json.dump(history_data, f, indent=4)
        
    epochs_range = range(1, epochs + 1)
    
    # Validation curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, multimodal_history['train_loss'], 'g--', label='Train Loss')
    plt.plot(epochs_range, multimodal_history['val_loss'], 'g-', label='Val Loss')
    plt.title('Multimodal Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, multimodal_history['val_acc'], 'r-', label='Val Acc')
    plt.plot(epochs_range, multimodal_history['val_auc'], 'b-', label='Val AUC')
    plt.title('Multimodal Validation Accuracy and AUC')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('processed/comparison_bai23.png')
    plt.close()
    
    # Attention chart
    plt.figure(figsize=(6, 5))
    modalities = ['Sequence', 'Image', 'Tabular']
    plt.bar(modalities, test_mean_attn, color=['blue', 'green', 'orange'])
    plt.title('Modality Attention Weights (Test Set)')
    plt.ylabel('Attention Weight')
    for i, val in enumerate(test_mean_attn):
        plt.text(i, val + 0.01, f"{val:.2%}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('processed/attention_weights.png')
    plt.close()
    
    print("Multimodal plots saved successfully.")

if __name__ == '__main__':
    main()
