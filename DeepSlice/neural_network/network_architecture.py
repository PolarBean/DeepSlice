"""PyTorch implementation of the DeepSlice network.

The network is the Keras Xception (with its 1000 class ImageNet top) followed by
three dense layers. Layers are named after their Keras counterparts so that the
original Keras ``.h5`` weight files can be loaded without conversion.
"""

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Keras BatchNormalization default
BN_EPSILON = 1e-3
# The residual branches of Xception are unnamed in Keras, so they get these auto-generated names
RESIDUAL_NAMES = [
    ("conv2d", "batch_normalization"),
    ("conv2d_1", "batch_normalization_1"),
    ("conv2d_2", "batch_normalization_2"),
    ("conv2d_3", "batch_normalization_3"),
]


class BatchNorm(nn.BatchNorm2d):
    """Batch normalisation with the Keras epsilon"""

    def __init__(self, channels):
        super().__init__(channels, eps=BN_EPSILON)


class SeparableConv2d(nn.Module):
    """Keras SeparableConv2D with a 3x3 kernel, 'same' padding and no bias."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def max_pool_same(x):
    """3x3 max pooling with stride 2 and TensorFlow 'same' padding.

    TensorFlow puts the extra padding on the bottom and right when the total
    padding is odd, which PyTorch's symmetric padding cannot express.
    """
    pads = []
    for size in (x.shape[3], x.shape[2]):
        out = -(-size // 2)
        total = max((out - 1) * 2 + 3 - size, 0)
        pads += [total // 2, total - total // 2]
    x = F.pad(x, pads, value=float("-inf"))
    return F.max_pool2d(x, 3, stride=2)


class DeepSliceNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        layers = {
            "block1_conv1": nn.Conv2d(3, 32, 3, stride=2, bias=False),
            "block1_conv1_bn": BatchNorm(32),
            "block1_conv2": nn.Conv2d(32, 64, 3, bias=False),
            "block1_conv2_bn": BatchNorm(64),
        }
        # entry flow blocks: (block, in channels, out channels of both separable convs)
        self.entry_blocks = [(2, 64, 128, 128), (3, 128, 256, 256), (4, 256, 728, 728)]
        # exit flow block 13 reuses the same structure with 728 then 1024 filters
        self.downsample_blocks = self.entry_blocks + [(13, 728, 728, 1024)]
        for (block, cin, c1, c2), (conv, bn) in zip(self.downsample_blocks, RESIDUAL_NAMES):
            layers[conv] = nn.Conv2d(cin, c2, 1, stride=2, bias=False)
            layers[bn] = BatchNorm(c2)
            layers[f"block{block}_sepconv1"] = SeparableConv2d(cin, c1)
            layers[f"block{block}_sepconv1_bn"] = BatchNorm(c1)
            layers[f"block{block}_sepconv2"] = SeparableConv2d(c1, c2)
            layers[f"block{block}_sepconv2_bn"] = BatchNorm(c2)
        for block in range(5, 13):
            for i in (1, 2, 3):
                layers[f"block{block}_sepconv{i}"] = SeparableConv2d(728, 728)
                layers[f"block{block}_sepconv{i}_bn"] = BatchNorm(728)
        layers["block14_sepconv1"] = SeparableConv2d(1024, 1536)
        layers["block14_sepconv1_bn"] = BatchNorm(1536)
        layers["block14_sepconv2"] = SeparableConv2d(1536, 2048)
        layers["block14_sepconv2_bn"] = BatchNorm(2048)
        layers["predictions"] = nn.Linear(2048, 1000)
        layers["dense"] = nn.Linear(1000, 256)
        layers["dense_1"] = nn.Linear(256, 256)
        layers["dense_2"] = nn.Linear(256, 9)
        self.layers = nn.ModuleDict(layers)

    def forward(self, x):
        """
        :param x: images of shape (batch, 299, 299, 3), as Keras expects them
        """
        L = self.layers
        x = x.permute(0, 3, 1, 2)
        x = F.relu(L["block1_conv1_bn"](L["block1_conv1"](x)))
        x = F.relu(L["block1_conv2_bn"](L["block1_conv2"](x)))

        def downsample_block(x, block, conv, bn):
            residual = L[bn](L[conv](x))
            # block 2 starts straight from the activated block 1 output
            if block != 2:
                x = F.relu(x)
            x = L[f"block{block}_sepconv1_bn"](L[f"block{block}_sepconv1"](x))
            x = F.relu(x)
            x = L[f"block{block}_sepconv2_bn"](L[f"block{block}_sepconv2"](x))
            return max_pool_same(x) + residual

        for (block, *_), (conv, bn) in zip(self.entry_blocks, RESIDUAL_NAMES):
            x = downsample_block(x, block, conv, bn)
        for block in range(5, 13):
            residual = x
            for i in (1, 2, 3):
                x = L[f"block{block}_sepconv{i}_bn"](L[f"block{block}_sepconv{i}"](F.relu(x)))
            x = x + residual
        x = downsample_block(x, 13, *RESIDUAL_NAMES[3])
        x = F.relu(L["block14_sepconv1_bn"](L["block14_sepconv1"](x)))
        x = F.relu(L["block14_sepconv2_bn"](L["block14_sepconv2"](x)))
        x = x.mean(dim=(2, 3))
        x = F.softmax(L["predictions"](x), dim=1)
        x = F.relu(L["dense"](x))
        x = F.relu(L["dense_1"](x))
        return L["dense_2"](x)


def _tensor(array):
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def _set_layer(layer, weights):
    """Copy Keras weights (a dict of weight name -> array) into a PyTorch layer"""
    with torch.no_grad():
        if isinstance(layer, BatchNorm):
            layer.weight.copy_(_tensor(weights["gamma"]))
            layer.bias.copy_(_tensor(weights["beta"]))
            layer.running_mean.copy_(_tensor(weights["moving_mean"]))
            layer.running_var.copy_(_tensor(weights["moving_variance"]))
        elif isinstance(layer, SeparableConv2d):
            # Keras: (h, w, in, 1) and (1, 1, in, out); PyTorch: (in, 1, h, w) and (out, in, 1, 1)
            layer.depthwise.weight.copy_(_tensor(np.transpose(weights["depthwise_kernel"], (2, 3, 0, 1))))
            layer.pointwise.weight.copy_(_tensor(np.transpose(weights["pointwise_kernel"], (3, 2, 0, 1))))
        elif isinstance(layer, nn.Conv2d):
            # Keras: (h, w, in, out); PyTorch: (out, in, h, w)
            layer.weight.copy_(_tensor(np.transpose(weights["kernel"], (3, 2, 0, 1))))
        elif isinstance(layer, nn.Linear):
            layer.weight.copy_(_tensor(np.transpose(weights["kernel"])))
            layer.bias.copy_(_tensor(weights["bias"]))


def _read_group(group):
    """Read a Keras h5 layer group into a dict of weight name (without ':0') -> array"""
    return {name.split(":")[0]: np.array(group[name]) for name in group}


def load_imagenet_classifier(model: DeepSliceNetwork, xception_weights: str):
    """Load the ImageNet top of Xception, which the DeepSlice weight files do not contain.

    The ImageNet file uses legacy Keras 1 layer names; its final dense layer is the classifier.
    """
    with h5py.File(xception_weights, "r") as f:
        group = f["dense_2"]
        weights = {"kernel": np.array(group["dense_2_W:0"]), "bias": np.array(group["dense_2_b:0"])}
    _set_layer(model.layers["predictions"], weights)
    return model


def load_deepslice_weights(model: DeepSliceNetwork, weights: str):
    """Load a DeepSlice Keras weight file (Xception base plus the three dense layers)"""
    with h5py.File(weights, "r") as f:
        loaded = set()
        for name in f["xception"]:
            if name in model.layers:
                _set_layer(model.layers[name], _read_group(f["xception"][name]))
                loaded.add(name)
        for name in ("dense", "dense_1", "dense_2"):
            _set_layer(model.layers[name], _read_group(f[name][name]))
            loaded.add(name)
    missing = set(model.layers) - loaded - {"predictions"}
    if missing:
        raise ValueError(f"{weights} has no weights for the layers: {sorted(missing)}")
    return model
