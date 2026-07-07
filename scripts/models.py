"""
Six architectures for GISLR sign classification, all taking input of shape
(batch, max_len, feature_dim) where feature_dim = 543 * (2 or 3), and
producing (batch, num_classes) logits.
"""
import math
import torch
import torch.nn as nn

try:
    from torchvision.models import efficientnet_b0
except ImportError:
    efficientnet_b0 = None


# ---------------------------------------------------------------------------
# LSTM / BiLSTM
# ---------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size) — final layer's last state
        return self.head(last_hidden)


class BiLSTMClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers * 2, batch, hidden). Concat final layer's
        # forward and backward states.
        forward_last = h_n[-2]
        backward_last = h_n[-1]
        combined = torch.cat([forward_last, backward_last], dim=-1)
        return self.head(combined)


# ---------------------------------------------------------------------------
# Unidirectional GRU (streaming-friendly — no future context used)
# ---------------------------------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        out, h_n = self.gru(x)
        last_hidden = h_n[-1]
        return self.head(last_hidden)


# ---------------------------------------------------------------------------
# 1D-CNN over the time axis
# ---------------------------------------------------------------------------
class CNN1DClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, channels=(256, 256, 512), dropout=0.3):
        super().__init__()
        layers = []
        in_ch = feature_dim
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        # x: (batch, time, features) -> conv1d wants (batch, channels, time)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Transformer encoder
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, d_model=256, nhead=8,
                 num_layers=4, dim_feedforward=512, dropout=0.3, max_len=128):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len + 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        batch = x.size(0)
        x = self.input_proj(x)
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


# ---------------------------------------------------------------------------
# Spectrogram-style model (2nd place approach): reshape the landmark
# sequence into a pseudo-image (points x time) and classify with a CNN
# image backbone (EfficientNet-B0), same way audio spectrograms are treated
# as images.
# ---------------------------------------------------------------------------
class SpectrogramCNNClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, img_size=(160, 80), pretrained=False):
        super().__init__()
        if efficientnet_b0 is None:
            raise ImportError("torchvision is required for SpectrogramCNNClassifier")

        self.img_size = img_size  # (height, width) = (points-ish, time-ish) after resize
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # Adapt first conv to accept 1 channel (coordinate channel), matching
        # the "treat landmarks-over-time as a single-channel spectrogram" idea.
        first_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            1, first_conv.out_channels, kernel_size=first_conv.kernel_size,
            stride=first_conv.stride, padding=first_conv.padding, bias=False,
        )
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, x):
        # x: (batch, time, feature_dim) where feature_dim = 543 * coords.
        # Collapse to a single "amplitude" channel per (landmark, coord) via
        # the vector norm, giving (batch, time, num_points) — analogous to a
        # magnitude spectrogram — then treat as a 1-channel image
        # (height=points, width=time).
        batch, T, F = x.shape
        coords = 2  # matches use_z=False default in the dataset; change if use_z=True
        num_points = F // coords
        x = x.view(batch, T, num_points, coords)
        x = torch.linalg.norm(x, dim=-1)  # (batch, T, num_points)
        x = x.permute(0, 2, 1).unsqueeze(1)  # (batch, 1, num_points, T) = (B, C, H, W)
        x = nn.functional.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        return self.backbone(x)


MODEL_REGISTRY = {
    "lstm": LSTMClassifier,
    "bilstm": BiLSTMClassifier,
    "gru": GRUClassifier,
    "cnn1d": CNN1DClassifier,
    "transformer": TransformerClassifier,
    "spectrogram": SpectrogramCNNClassifier,
}
