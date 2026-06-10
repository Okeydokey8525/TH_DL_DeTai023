import os
import pandas as pd
import numpy as np
import wfdb
import ast
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from sklearn.preprocessing import StandardScaler

class ECGPreprocessingPipeline:
    def __init__(self, data_path, sampling_rate=100):
        # Hướng đường dẫn ra thư mục cha chứa dữ liệu thô
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        self.num_classes = len(self.classes)
        self.scaler_tabular = StandardScaler()
        
    def load_metadata(self):
        print("Loading CSV metadata...")
        self.df = pd.read_csv(os.path.join(self.data_path, 'ptbxl_database.csv'), index_col='ecg_id')
        self.df.scp_codes = self.df.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        agg_df = pd.read_csv(os.path.join(self.data_path, 'scp_statements.csv'), index_col=0)
        self.diag_map = agg_df[agg_df.diagnostic == 1].diagnostic_class.to_dict()
        
        def aggregate_diagnostic(scp_dict):
            tmp = set()
            for key in scp_dict.keys():
                if key in self.diag_map:
                    tmp.add(self.diag_map[key])
            return list(tmp)
            
        self.df['diagnostic_superclass'] = self.df.scp_codes.apply(aggregate_diagnostic)
        
        # Lọc bỏ các bản ghi không có nhãn thuộc 5 lớp chính
        self.df = self.df[self.df['diagnostic_superclass'].apply(len) > 0].copy()
        
        def make_multihot(labels):
            vec = np.zeros(self.num_classes, dtype=np.float32)
            for lbl in labels:
                if lbl in self.classes:
                    idx = self.classes.index(lbl)
                    vec[idx] = 1.0
            return vec
            
        self.labels = np.array([make_multihot(l) for l in self.df['diagnostic_superclass']])
        print(f"Total clean records for Bai 23: {len(self.df)}")
        return self.df
        
    def process_tabular_data(self):
        print("Processing tabular metadata...")
        tabular_df = self.df[['age', 'sex', 'height', 'weight']].copy()
        tabular_df['sex'] = tabular_df['sex'].fillna(0).astype(np.float32)
        
        tabular_df['age'] = tabular_df['age'].fillna(tabular_df['age'].median())
        tabular_df['height'] = tabular_df['height'].fillna(tabular_df['height'].median())
        tabular_df['weight'] = tabular_df['weight'].fillna(tabular_df['weight'].median())
        
        self.tabular_features = self.scaler_tabular.fit_transform(tabular_df).astype(np.float32)
        return self.tabular_features

    def load_raw_signals(self):
        print("Loading raw ECG signals from disk (this might take 1-2 minutes)...")
        signals = []
        filenames = self.df.filename_lr if self.sampling_rate == 100 else self.df.filename_hr
        
        for i, f in enumerate(filenames):
            full_path = os.path.join(self.data_path, f)
            signal, meta = wfdb.rdsamp(full_path)
            signals.append(signal)
            
            if (i+1) % 5000 == 0:
                print(f"Loaded {i+1}/{len(filenames)} signals.")
                
        signals = np.array(signals, dtype=np.float32)
        
        print("Normalizing signals (Z-score)...")
        for i in range(len(signals)):
            mean = signals[i].mean(axis=0, keepdims=True)
            std = signals[i].std(axis=0, keepdims=True) + 1e-8
            signals[i] = (signals[i] - mean) / std
            
        self.signals = signals
        return self.signals

    @staticmethod
    def render_ecg_to_image(signal, img_size=128):
        img = Image.new('L', (img_size, img_size), color=255)
        draw = ImageDraw.Draw(img)
        num_leads = 12
        lead_height = img_size / num_leads
        xs = np.linspace(0, len(signal) - 1, img_size).astype(int)
        
        for i in range(num_leads):
            lead_signal = signal[xs, i]
            lead_signal = np.clip(lead_signal, -2.5, 2.5)
            ys = (i + 0.5) * lead_height - lead_signal * (lead_height / 5.0)
            ys = np.clip(ys, i * lead_height, (i + 1) * lead_height - 1)
            
            for col in range(img_size - 1):
                draw.line((col, ys[col], col + 1, ys[col + 1]), fill=0, width=1)
                
        return np.array(img, dtype=np.float32) / 255.0

    def generate_all_images(self):
        print("Rendering 2D ECG image representations...")
        images = []
        for i in range(len(self.signals)):
            img = self.render_ecg_to_image(self.signals[i])
            images.append(np.expand_dims(img, axis=0))
            if (i+1) % 5000 == 0:
                print(f"Rendered {i+1}/{len(self.signals)} images.")
        self.images = np.array(images, dtype=np.float32)
        return self.images

    def save_processed_data(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print("Saving processed datasets...")
        
        folds = self.df.strat_fold.values
        
        train_idx = np.where((folds != 9) & (folds != 10))[0]
        val_idx = np.where(folds == 9)[0]
        test_idx = np.where(folds == 10)[0]
        
        np.savez_compressed(
            os.path.join(output_dir, 'ptbxl_processed_bai23.npz'),
            x_train_seq=self.signals[train_idx],
            x_train_num=self.tabular_features[train_idx],
            x_train_img=self.images[train_idx],
            y_train=self.labels[train_idx],
            x_val_seq=self.signals[val_idx],
            x_val_num=self.tabular_features[val_idx],
            x_val_img=self.images[val_idx],
            y_val=self.labels[val_idx],
            x_test_seq=self.signals[test_idx],
            x_test_num=self.tabular_features[test_idx],
            x_test_img=self.images[test_idx],
            y_test=self.labels[test_idx],
            test_ecg_ids=self.df.index.values[test_idx]
        )
        print("Data processing and saving complete for Bai 23!")

class MultimodalECGDataset(Dataset):
    def __init__(self, signals, images, tabulars, labels):
        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.images = torch.tensor(images, dtype=torch.float32)
        self.tabulars = torch.tensor(tabulars, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.signals)
        
    def __getitem__(self, idx):
        signal = self.signals[idx].permute(1, 0)
        image = self.images[idx]
        tabular = self.tabulars[idx]
        label = self.labels[idx]
        return signal, image, tabular, label

if __name__ == '__main__':
    raw_data_path = '../'
    pipeline = ECGPreprocessingPipeline(raw_data_path)
    pipeline.load_metadata()
    pipeline.process_tabular_data()
    pipeline.load_raw_signals()
    pipeline.generate_all_images()
    pipeline.save_processed_data('processed')
