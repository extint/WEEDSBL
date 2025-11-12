# models/dannet_from_mannet.py
import torch
import torch.nn as nn
from models.mannet import LightMANet  # uses your existing model

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = float(lambda_)
    def forward(self, x):
        return GradientReversal.apply(x, self.lambda_)

class DomainClassifier(nn.Module):
    def __init__(self, in_ch, hidden=64, num_domains=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # nn.Linear(in_ch, hidden),
            # nn.BatchNorm1d(hidden),
            # nn.ReLU(inplace=True),
            # nn.Linear(hidden, num_domains)
            nn.Linear(in_ch, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        return self.net(x)

class DANNFromMANet(nn.Module):
    """
    Training: returns (seg_logits, domain_logits)
    Eval/Inference: returns seg_logits
    """
    def __init__(self, in_channels=4, base_ch=16, out_channels=1, grl_lambda=1.0):
        super().__init__()
        # Reuse your LightMANet instance and its submodules directly
        self.net = LightMANet(in_channels=in_channels, num_classes=out_channels, base_ch=base_ch)
        # Domain head attached to bottleneck (layer4 output channels = base_ch*8)
        self.grl = GradientReversalLayer(lambda_=grl_lambda)
        self.domain_disc = DomainClassifier(in_ch=base_ch * 8, hidden=64, num_domains=2)

    def encode_to_bottleneck(self, x):
        # Traverse explicit stages from your LightMANet to get bottleneck pre-decoder
        x = self.net.conv1(x)      # H/4
        x = self.net.layer1(x)     # H/4
        x = self.net.layer2(x)     # H/8
        x = self.net.layer3(x)     # H/16
        x = self.net.layer4(x)     # H/32  <- bottleneck
        return x

    def decode_to_logits(self, bottleneck):
        x = self.net.decoder(bottleneck)
        logits = self.net.final(x)
        return logits

    def forward(self, x):
        feats = self.encode_to_bottleneck(x)
        seg_logits = self.decode_to_logits(feats)
        if self.training:
            dom_logits = self.domain_disc(self.grl(feats))
            return seg_logits, dom_logits
        else:
            return seg_logits
