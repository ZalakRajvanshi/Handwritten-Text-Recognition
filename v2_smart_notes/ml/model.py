import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    CRNN for single-line handwriting OCR
    Input: (B, 1, 32, W)
    """

    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 32 -> 16

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 16 -> 8

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU()
        )

        # IMPORTANT: 256 * 8 = 2048 (must match training)
        self.rnn = nn.LSTM(
            input_size=2048,
            hidden_size=256,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)                  # (B, 256, 8, W)
        b, c, h, w = x.size()

        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(b, w, c * h)  # (B, W, 2048)

        x, _ = self.rnn(x)               # (B, W, 512)
        x = self.fc(x)                   # (B, W, num_classes)

        return x
