import torch


class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = torch.nn.Sequential(
            # 输入输出通道 11*11卷积核 4步长
            # input 3通道 224*224 -> 96通道 55*55 多少通道由多少卷积核决定
            torch.nn.Conv2d(1, 96, 11, 4),
            torch.nn.ReLU(),
            # 3*3最大池化核 2步长 96通道 55*55 -> 27*27
            torch.nn.MaxPool2d(3, 2),
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),
            torch.nn.Conv2d(256, 384, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384, 384, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        self.fc = torch.nn.Sequential(
            # 所有输出全算上
            torch.nn.Linear(256 * 5 * 5, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
