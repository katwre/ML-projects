import torch
import torch.nn as nn
from torchinfo import summary
from typing import Dict, Any



##### CNNs:

class CNN(nn.Module):
    def __init__(
        self,
        input_size: int = 224,         # works for 224, 64, etc.
        channel_numbers: int = 1,
        kernel_size: int = 16,
        stride: int = 4,
        use_leakyrelu: bool = False,   # optional
    ):
        super().__init__()
        act = nn.LeakyReLU(0.1, inplace=True) if use_leakyrelu else nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_numbers,
                out_channels=2,
                kernel_size=kernel_size,
                stride=stride,
            ),
            act,
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5, stride=1),
            act,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()

        # Dynamically infer the flattened feature size from input_size ---
        with torch.no_grad():
            dummy = torch.zeros(1, channel_numbers, input_size, input_size)
            feat_dim = self._feat_dim_after_conv(dummy)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 16),
            nn.Linear(16, 8),
            nn.Linear(8, 1),
            nn.Sigmoid(),  # we're using BCELoss in training
        )

    def _feat_dim_after_conv(self, x: torch.Tensor) -> int:
        h = self.conv(x)
        return h.view(h.size(0), -1).size(1)

    def forward(self, x):
        out = self.conv(x)
        out = self.flatten(out)
        out = self.classifier(out)
        return out


def create_model(device: torch.device, input_size: int = 224, **kwargs) -> nn.Module:
    """
    kwargs are passed to CNN (e.g., use_leakyrelu=True, kernel_size=16, stride=4, ...)
    """
    model = CNN(
        input_size=input_size,
        channel_numbers=1,
        kernel_size=16,
        stride=4,
        **kwargs,
    ).to(device)
    return model

def create_model(
    device: torch.device,
    input_size: int = 224,
    channel_numbers: int = 1,
    kernel_size: int = 16,
    stride: int = 4,
    use_leakyrelu: bool = False,
) -> nn.Module:
    """
    kwargs are passed to CNN (e.g., use_leakyrelu=True, kernel_size=16, stride=4, ...)
    """
    model = CNN(
        input_size=input_size,
        channel_numbers=channel_numbers,
        kernel_size=kernel_size,
        stride=stride,
        use_leakyrelu=use_leakyrelu,
    ).to(device)
    return model


def describe_model(model: torch.nn.Module, input_size=(1, 1, 224, 224)):
    """Pretty-print model architecture and parameter summary."""
    print("\n" + "="*60)
    print("[INFO] Model architecture:\n")
    print(model)
    try:
        print("\n[INFO] Detailed summary:")
        summary(model, input_size=input_size)
    except Exception as e:
        print(f"[WARN] Could not display detailed summary ({e})")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    print("="*60 + "\n", flush=True)



### What each layer does on the example of input images of 224x224? (not used)

## Input:
# Shape: [1, 1, 224, 224] (batch, channel, height, width)
# 1 channel = grayscale X-ray

## Convolution layer 1 -> nn.Conv2d(1, 2, kernel_size=16, stride=4) :
# Takes 1 input channel (the image)
# Produces 2 output feature maps
# Each filter looks at a 16×16 patch, moving 4 pixels at a time
# Detects basic edges, bright/dark blobs, textures
# Output shape ≈ [1, 2, 53, 53] (use formula: (224−16)/4 + 1 = 53)

## ReLU
#Adds non-linearity → turns negative values into 0
#so the model can learn nonlinear features.

## MaxPool2d
# Takes 2×2 patches and keeps only the largest value → reduces spatial size by 2×
# Helps the network become invariant to small translations (i.e., it doesn’t care if a lung texture moves 1–2 pixels).
# Output shape ≈ [1, 2, 26, 26]

## Convolution layer 2 -> nn.Conv2d(2, 4, kernel_size=5, stride=1):
# Takes 2 feature maps from before
# Produces 4 new feature maps
# Learns combinations of low-level features (edges → corners, patterns)
# Output shape ≈ [1, 4, 22, 22]

## Flatten
#Turns the 4 feature maps of shape [4, 11, 11] into a single vector
# → 4 × 11 × 11 = 484 features

## Fully Connected Classifier
# A small multilayer perceptron that:
# - Combines learned spatial features into class-level information
# - Outputs a single probability (after Sigmoid) for PNEUMONIA vs NORMAL

### CNN that works ony on the 224x224 images
class CNN_hardcoded224x224(nn.Module): 
    # (not used)
    def __init__(
        self,
        input_size: int = 224,
        channel_numbers: int = 1,
        kernel_size: int = 16,
        stride: int = 4,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_numbers,
                out_channels=2,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()

        # keep your fixed flatten size (based on 224 input + conv config)
        flatten_output_dim = 484

        self.classifier = nn.Sequential(
            nn.Linear(flatten_output_dim, 16),
            nn.Linear(16, 8),
            nn.Linear(8, 1),
            nn.Sigmoid(),  # we'll use BCELoss
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.flatten(out)
        out = self.classifier(out)
        return out


def create_model_hardcoded224x224(device: torch.device) -> nn.Module:
    # (not used)
    model = CNN_hardcoded224x224(
        input_size=224,
        channel_numbers=1,
        kernel_size=16,
        stride=4,
    ).to(device)
    return model

##### Transfer learning (not used at the moment)

# -----------------------
#  Helpers for transfer learning backbones
# -----------------------

def _adapt_first_conv_to_grayscale(m: nn.Module) -> None:
    """If model expects RGB, make it 1-channel by averaging RGB weights."""
    if hasattr(m, "conv1") and isinstance(m.conv1, nn.Conv2d) and m.conv1.in_channels == 3:
        old = m.conv1
        new = nn.Conv2d(
            1, old.out_channels, kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding, bias=(old.bias is not None)
        )
        with torch.no_grad():
            new.weight[:] = old.weight.mean(dim=1, keepdim=True)
            if old.bias is not None:
                new.bias[:] = old.bias
        m.conv1 = new


def _replace_classifier_with_sigmoid_head(m: nn.Module, in_features: int) -> None:
    head = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())
    if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
        m.fc = head
        return
    if hasattr(m, "classifier"):
        if isinstance(m.classifier, nn.Sequential) and isinstance(m.classifier[-1], nn.Linear):
            m.classifier[-1] = head
            return
        if isinstance(m.classifier, nn.Linear):
            m.classifier = head
            return
    raise ValueError("Unsupported classifier structure for this backbone.")


def _freeze_all_but_head(m: nn.Module) -> None:
    for n, p in m.named_parameters():
        if "fc" in n or "classifier" in n:
            continue
        p.requires_grad = False


# -----------------------
#  Build TL model (not used at the momenbt)
# -----------------------

def build_transfer_model(cfg: dict, device: torch.device) -> nn.Module:
    """Create and configure a pretrained CNN backbone for binary classification."""
    from torchvision import models

    backbone = cfg.get("backbone", "resnet50").lower()
    pretrained = bool(cfg.get("pretrained", True))
    freeze = bool(cfg.get("freeze_backbone", True))

    # --- load backbone ---
    if backbone == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        in_features = model.fc.in_features

    elif backbone == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        in_features = model.fc.in_features

    elif hasattr(models, "efficientnet_b0") and backbone == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        in_features = model.classifier[-1].in_features

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # --- modify for grayscale + binary head ---
    _adapt_first_conv_to_grayscale(model)
    _replace_classifier_with_sigmoid_head(model, in_features)

    # --- optionally freeze backbone ---
    if freeze:
        _freeze_all_but_head(model)

    return model.to(device)


##### CNN and Transfer learning functions:

# =========================
# Unified builder (not used at the moment)
# =========================
def build_model(cfg: Dict[str, Any],
                device: torch.device, 
                **kwargs) -> nn.Module:
    """
    cfg example:
    model:
      kind: cnn            # 'cnn' or 'tl'
      # cnn-only:
      input_size: 225 # or 64
      kernel_size: 16
      stride: 4
      use_leakyrelu: false
      # tl-only:
      backbone: resnet50 # or other resnet models
      pretrained: true
      freeze_backbone: true
    """

    kind = cfg.get("kind", "cnn").lower()

    if kind == "cnn":
        input_size = int(cfg.get("input_size", 224)) 
        ks = int(cfg.get("kernel_size", 16))
        st = int(cfg.get("stride", 4))
        use_lrelu = bool(cfg.get("use_leakyrelu", False))
        return create_model(
            device=device,
            input_size=input_size,
            kernel_size=ks,
            stride=st,
            use_leakyrelu=use_lrelu,
            **kwargs
        )

    elif kind == "tl":  
        return build_transfer_model(cfg, device)
    
    else:
        raise ValueError(f"Unknown model.kind: {kind}")

