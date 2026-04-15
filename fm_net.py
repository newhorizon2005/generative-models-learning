import torch

class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, condition_size):
        super().__init__()
        self.downconv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels + condition_size, out_channels=out_channels, kernel_size=3,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)

    def add_condition(self, tensor, cond_emb):
        cond_emb = cond_emb.view(cond_emb.size(0), cond_emb.size(1), 1, 1)
        return torch.concat([tensor, cond_emb.expand(-1, -1, tensor.size(2), tensor.size(3))], dim=1)

    def forward(self, x, condition):
        x = self.downconv(self.add_condition(x, condition))
        return x, self.maxpool(x)


class UpSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, condition_size):
        super().__init__()
        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=in_channels + condition_size, out_channels=out_channels, kernel_size=4,
                                     stride=2, padding=1),
            torch.nn.ReLU(),
        )
        self.upconv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels + condition_size, out_channels=out_channels, kernel_size=3,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )

    def add_condition(self, tensor, cond_emb):
        cond_emb = cond_emb.view(cond_emb.size(0), cond_emb.size(1), 1, 1)
        return torch.concat([tensor, cond_emb.expand(-1, -1, tensor.size(2), tensor.size(3))], dim=1)

    def forward(self, x, redidual_x, condition):
        x = self.deconv(self.add_condition(x, condition))
        x = torch.concat([x, redidual_x], dim=1)
        x = self.upconv(self.add_condition(x, condition))
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # condition
        self.label_emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=16)
        self.t_emb = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=16),
        )
        self.condition_size = 32

        self.down0 = DownSample(in_channels=1, out_channels=64, condition_size=self.condition_size)
        self.down1 = DownSample(in_channels=64, out_channels=128, condition_size=self.condition_size)
        self.down2 = DownSample(in_channels=128, out_channels=256, condition_size=self.condition_size)

        self.up0 = UpSample(in_channels=256, out_channels=128, condition_size=self.condition_size)
        self.up1 = UpSample(in_channels=128, out_channels=64, condition_size=self.condition_size)

        self.output_conv = torch.nn.Conv2d(in_channels=64 + self.condition_size, out_channels=1, kernel_size=3,
                                           padding=1)

    def add_condition(self, tensor, cond_emb):
        cond_emb = cond_emb.view(cond_emb.size(0), cond_emb.size(1), 1, 1)
        return torch.concat([tensor, cond_emb.expand(-1, -1, tensor.size(2), tensor.size(3))], dim=1)

    def forward(self, x, t, label):
        cond_emb = torch.concat((self.label_emb(label), self.t_emb(t.unsqueeze(1))), dim=1)

        x0, x = self.down0(x, cond_emb)  # torch.Size([128, 64, 28, 28]) torch.Size([128, 64, 14, 14])
        x1, x = self.down1(x, cond_emb)  # torch.Size([128, 128, 28, 28]) torch.Size([128, 128, 14, 14])
        x, _ = self.down2(x, cond_emb)  # torch.Size([128, 256, 7, 7])
        x = self.up0(x, x1, cond_emb)  # torch.Size([128, 128, 14, 14])
        x = self.up1(x, x0, cond_emb)  # torch.Size([128, 64, 28, 28])
        return self.output_conv(self.add_condition(x, cond_emb))  # torch.Size([128, 1, 28, 28])