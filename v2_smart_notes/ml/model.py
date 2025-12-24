import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes):
        super(CRNN, self).__init__()

        # ---------------- CNN ----------------
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # H/2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # H/4

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # H/8

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # H/16
        )

        # ---------------- RNN ----------------
        self.lstm1 = nn.LSTM(1024, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)

        # ---------------- Classifier ----------------
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        conv = self.cnn(x)

        # (B, C, H, W) -> (B, W, C*H)
        b, c, h, w = conv.size()
        conv = conv.permute(0, 3, 1, 2)
        conv = conv.contiguous().view(b, w, c * h)

        # RNN
        rnn_out, _ = self.lstm1(conv)
        rnn_out, _ = self.lstm2(rnn_out)

        output = self.fc(rnn_out)
        return output
