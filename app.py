import os
import json
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

# Import Models and Datasets from current directory
from models import MultimodalAttentionNetwork

app = Flask(__name__)

DATA_DIR = 'processed'
DATA_FILE = os.path.join(DATA_DIR, 'ptbxl_processed_bai23.npz')

dataset = None
model_multimodal = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_resources():
    global dataset, model_multimodal
    
    if os.path.exists(DATA_FILE):
        try:
            dataset = np.load(DATA_FILE)
            print("Successfully loaded Bài 23 test dataset.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            
    model_path = os.path.join(DATA_DIR, 'multimodal_ecg.pt')
    if os.path.exists(model_path):
        try:
            model_multimodal = MultimodalAttentionNetwork()
            model_multimodal.load_state_dict(torch.load(model_path, map_location=device))
            model_multimodal.to(device)
            model_multimodal.eval()
            print("Successfully loaded Bài 23 Multimodal Model.")
        except Exception as e:
            print(f"Error loading Multimodal Model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    load_resources()
    has_data = dataset is not None
    has_model = model_multimodal is not None
    return jsonify({
        'status': 'ready' if (has_data and has_model) else 'mock_mode',
        'has_data': has_data,
        'has_model': has_model,
        'device': str(device)
    })

@app.route('/api/samples')
def get_samples():
    load_resources()
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    if dataset is None:
        mock_samples = []
        for i in range(15):
            mock_samples.append({
                'index': i,
                'ecg_id': 20000 + i,
                'sex': 'Male' if i % 2 == 0 else 'Female',
                'actual': [classes[i % 5]]
            })
        return jsonify(mock_samples)
    
    test_ecg_ids = dataset['test_ecg_ids']
    y_test = dataset['y_test']
    
    samples = []
    num_samples = min(20, len(test_ecg_ids))
    for idx in range(num_samples):
        # Trích xuất sex thực tế từ dataset
        sex_val = dataset['x_test_num'][idx][1]
        sex_str = "Male" if sex_val > 0 else "Female"
        
        labels_indices = np.where(y_test[idx] == 1.0)[0]
        actual_labels = [classes[li] for li in labels_indices]
        
        samples.append({
            'index': idx,
            'ecg_id': int(test_ecg_ids[idx]),
            'sex': sex_str,
            'actual': actual_labels
        })
    return jsonify(samples)

@app.route('/api/ecg/<int:idx>')
def get_ecg_details(idx):
    load_resources()
    
    if dataset is None:
        t = np.linspace(0, 10, 1000)
        signal = np.zeros((1000, 12))
        for lead in range(12):
            signal[:, lead] = np.sin(t * (lead + 1)) * np.exp(-t/5)
            for beat in range(1, 10):
                signal[beat*100:beat*100+10, lead] += 1.5 * (lead % 2 * 2 - 1)
        return jsonify({
            'ecg_id': 20000 + idx,
            'age': 55,
            'sex': 'Male' if idx % 2 == 0 else 'Female',
            'height': 170.0,
            'weight': 65.0,
            'signal': signal.tolist(),
            'image_available': False
        })
        
    x_seq = dataset['x_test_seq'][idx]
    x_num = dataset['x_test_num'][idx]
    
    # Khôi phục thông số bệnh nhân từ chuẩn hóa
    age = int(x_num[0] * 18 + 60)
    sex = "Male" if x_num[1] > 0 else "Female"
    height = float(x_num[2] * 11 + 166)
    weight = float(x_num[3] * 15 + 70)
    
    return jsonify({
        'ecg_id': int(dataset['test_ecg_ids'][idx]),
        'age': max(5, min(110, age)),
        'sex': sex,
        'height': round(max(100, min(220, height)), 1),
        'weight': round(max(30, min(180, weight)), 1),
        'signal': x_seq.tolist(),
        'image_available': True
    })

@app.route('/api/predict/<int:idx>')
def run_predict(idx):
    load_resources()
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    if model_multimodal is None or dataset is None:
        np.random.seed(idx)
        probs_bai23 = np.random.dirichlet(np.ones(5) * 3)[0]
        attn_weights = np.random.dirichlet([5, 3, 2])[0]
        pred_bai23 = {classes[i]: float(probs_bai23[i]) for i in range(5)}
        
        return jsonify({
            'mode': 'demo_mock',
            'predict_bai23': pred_bai23,
            'attention': {
                'sequence': float(attn_weights[0]),
                'image': float(attn_weights[1]),
                'tabular': float(attn_weights[2])
            }
        })
        
    x_seq_val = dataset['x_test_seq'][idx]
    x_img_val = dataset['x_test_img'][idx]
    x_num_val = dataset['x_test_num'][idx]
    
    tensor_seq = torch.tensor(x_seq_val, dtype=torch.float32).permute(1, 0).unsqueeze(0).to(device)
    tensor_img = torch.tensor(x_img_val, dtype=torch.float32).unsqueeze(0).to(device)
    tensor_num = torch.tensor(x_num_val, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits_bai23, attn_w = model_multimodal(tensor_seq, tensor_img, tensor_num)
        probs_bai23 = torch.sigmoid(logits_bai23).squeeze(0).cpu().numpy()
        attn_w = attn_w.squeeze(0).cpu().numpy()
        
    pred_bai23 = {classes[i]: float(probs_bai23[i]) for i in range(5)}
    
    return jsonify({
        'mode': 'pytorch_inference',
        'predict_bai23': pred_bai23,
        'attention': {
            'sequence': float(attn_w[0]),
            'image': float(attn_w[1]),
            'tabular': float(attn_w[2])
        }
    })

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    load_resources()
    print("Flask app for Bai 23 starting on http://127.0.0.1:5002/")
    app.run(debug=True, port=5002)
