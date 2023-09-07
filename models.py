import torch
import torch.nn as nn
from resnet import resnet18, resnet50
import torch.nn.functional as F
from scipy.stats import norm


class GazeRes(nn.Module):
    def __init__(self, backbone = "res18", drop_p=0.5):
        super(GazeRes, self).__init__()
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        if backbone == "res18":
            self.base_model = resnet18(pretrained=True)
        elif backbone == "res50":
            self.base_model = resnet50(pretrained=True)


        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)

        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer = nn.Linear(self.img_feature_dim, 2)
        self.drop = nn.Dropout(drop_p)

    def forward(self, x_in):
        base_out = self.base_model(x_in["face"])
        base_out = torch.flatten(base_out, start_dim=1)
        output = self.drop(base_out)
        output = self.last_layer(output)
        angular_output = output[:, :2]

        return angular_output, base_out


class UncertaintyLoss(nn.Module):
    def __index__(self):
        sum(UncertaintyLoss, self).__init__()
    def forward(self, gaze, gaze_ema):
        assert gaze.shape == gaze_ema.shape
        std = torch.std(gaze, dim=2).reshape(-1, 2, 1)
        return torch.mean(std)
    def forward(self, gaze, gaze_ema, significant=1, std_alpha=0.2, gamma=0.01):
        assert gaze.shape == gaze_ema.shape
        std = torch.std(gaze, dim=2).reshape(-1, 2, 1)
        return torch.mean(std)

class UncertaintyPseudoLabelLoss(nn.Module):
    def __init__(self, lamda_pseudo = 0.5):
        super(UncertaintyPseudoLabelLoss, self).__init__()
        self.lamda_pseudo = lamda_pseudo 
    def forward(self, gaze, gaze_ema):
        assert gaze.shape == gaze_ema.shape
        std = torch.std(gaze, dim=2).reshape(-1, 2, 1)
        mean = torch.mean(gaze_ema, dim=2).reshape(-1, 2, 1)
        return torch.mean(std) + self.lamda_pseudo * torch.mean(torch.abs(gaze - mean))

class UncertaintyWPseudoLabelLoss(nn.Module):
    def __init__(self, lamda_pseudo = 0.5):
        super(UncertaintyWPseudoLabelLoss, self).__init__()
        self.lamda_pseudo = lamda_pseudo 
    def forward(self, gaze, gaze_ema):
        assert gaze.shape == gaze_ema.shape
        std = torch.std(gaze, dim=2).reshape(-1, 2, 1)
        mean = torch.mean(gaze_ema, dim=2).reshape(-1, 2, 1)
        return torch.mean(std) + self.lamda_pseudo * torch.mean(torch.abs(gaze - mean) / std.detach())


class WeightedPseudoLabelLoss(nn.Module):
    def __init__(self, lamda_pseudo = 0.5):
        super(WeightedPseudoLabelLoss, self).__init__()
        # self.lamda_pseudo = lamda_pseudo 
    def forward(self, gaze, gaze_ema):
        assert gaze.shape == gaze_ema.shape
        std = torch.std(gaze, dim=2).reshape(-1, 2, 1)
        mean = torch.mean(gaze_ema, dim=2).reshape(-1, 2, 1)
        return torch.mean(torch.abs(gaze - mean) / std.detach())

class PseudoLabelLoss(nn.Module):
    def __init__(self, lamda_pseudo = 0.5):
        super(PseudoLabelLoss, self).__init__()
        # self.lamda_pseudo = lamda_pseudo 
    def forward(self, gaze, gaze_ema):
        assert gaze.shape == gaze_ema.shape
        mean = torch.mean(gaze_ema, dim=2).reshape(-1, 2, 1)
        return torch.mean(torch.abs(gaze - mean))