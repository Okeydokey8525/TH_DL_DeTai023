import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConvBlock, self).__init__()
        assert out_channels % 3 == 0, "out_channels phải chia hết cho 3"
        branch_channels = out_channels // 3
        
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat([out1, out2, out3], dim=1)


class MultiScaleECGModel(nn.Module):
    """
    Nhánh xử lý Tín hiệu ECG của Bài 23.
    """
    def __init__(self, in_channels=12, num_classes=5):
        super(MultiScaleECGModel, self).__init__()
        
        self.block1 = nn.Sequential(
            MultiScaleConvBlock(in_channels, 96),
            nn.MaxPool1d(2)
        )
        self.block2 = nn.Sequential(
            MultiScaleConvBlock(96, 192),
            nn.MaxPool1d(2)
        )
        self.block3 = nn.Sequential(
            MultiScaleConvBlock(192, 384),
            nn.MaxPool1d(2)
        )
        
        self.bilstm = nn.LSTM(input_size=384, hidden_size=64, num_layers=2,
                             batch_first=True, bidirectional=True, dropout=0.3)
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def extract_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.bilstm(x)
        fused_features = torch.mean(lstm_out, dim=1)
        return fused_features

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.fc(features)
        return logits


class ImageBranch2DCNN(nn.Module):
    """
    Nhánh xử lý Ảnh đồ thị ECG (2D-CNN).
    Đầu vào: (Batch, Channels=1, Height=128, Width=128)
    Đầu ra: (Batch, Features=128)
    """
    def __init__(self):
        super(ImageBranch2DCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # (Batch, 16, 64, 64)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # (Batch, 32, 32, 32)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # (Batch, 64, 16, 16)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # (Batch, 128, 8, 8)
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)


class TabularBranchMLP(nn.Module):
    """
    Nhánh xử lý đặc trưng Số nhân khẩu học (Tuổi, Giới tính, Chiều cao, Cân nặng).
    Đầu vào: (Batch, Features=4)
    Đầu ra: (Batch, Features=64)
    """
    def __init__(self):
        super(TabularBranchMLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        return self.mlp(x)


class ModalityAttentionFusion(nn.Module):
    """
    Lớp hợp nhất đặc trưng dựa trên cơ chế Modality-level Attention.
    """
    def __init__(self, seq_dim=128, img_dim=128, tab_dim=64, projection_dim=64):
        super(ModalityAttentionFusion, self).__init__()
        
        self.proj_seq = nn.Linear(seq_dim, projection_dim)
        self.proj_img = nn.Linear(img_dim, projection_dim)
        self.proj_tab = nn.Linear(tab_dim, projection_dim)
        
        self.attention_score = nn.Sequential(
            nn.Linear(projection_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
    def forward(self, seq_feat, img_feat, tab_feat):
        proj_seq = self.proj_seq(seq_feat) # (batch, 64)
        proj_img = self.proj_img(img_feat) # (batch, 64)
        proj_tab = self.proj_tab(tab_feat) # (batch, 64)
        
        score_seq = self.attention_score(proj_seq) # (batch, 1)
        score_img = self.attention_score(proj_img) # (batch, 1)
        score_tab = self.attention_score(proj_tab) # (batch, 1)
        
        scores = torch.cat([score_seq, score_img, score_tab], dim=1) # (batch, 3)
        attn_weights = F.softmax(scores, dim=1) # (batch, 3)
        
        alpha_seq = attn_weights[:, 0:1]
        alpha_img = attn_weights[:, 1:2]
        alpha_tab = attn_weights[:, 2:3]
        
        fused_features = (alpha_seq * proj_seq) + (alpha_img * proj_img) + (alpha_tab * proj_tab)
        return fused_features, attn_weights


class MultimodalAttentionNetwork(nn.Module):
    """
    Hệ thống phân loại đa nguồn (Bài 23).
    """
    def __init__(self, num_classes=5):
        super(MultimodalAttentionNetwork, self).__init__()
        
        self.seq_branch = MultiScaleECGModel(in_channels=12, num_classes=num_classes)
        self.img_branch = ImageBranch2DCNN()
        self.tab_branch = TabularBranchMLP()
        
        self.fusion = ModalityAttentionFusion(seq_dim=128, img_dim=128, tab_dim=64, projection_dim=64)
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x_seq, x_img, x_tab):
        seq_feat = self.seq_branch.extract_features(x_seq)
        img_feat = self.img_branch(x_img)
        tab_feat = self.tab_branch(x_tab)
        
        fused_feat, attn_weights = self.fusion(seq_feat, img_feat, tab_feat)
        logits = self.classifier(fused_feat)
        return logits, attn_weights
