"""U-Net that corrects a simulated path gain map.

The structure follows the U-Net of Geo2SigMap (arXiv:2312.14303) with these changes:
3 levels and 32 base channels (from the configuration), GroupNorm (arXiv:1803.08494) in
place of BatchNorm, bilinear up-sampling in place of a transposed convolution (Odena et al.,
"Deconvolution and Checkerboard Artifacts", Distill 2016), and an output layer with zero
initial weights.
"""

import torch
from torch import nn
from torch.nn import functional


class _Block(nn.Sequential):
    """Two 3x3 convolutions, each with GroupNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Encoder-decoder network with skip connections that gives one map for each image.

    Args:
        in_channels: Number of input channels.
        base_channels: Number of channels of the first level. Each lower level has two
            times the channels of the level above it.
        depth: Number of down-sampling steps.
        groups: Number of groups of each GroupNorm layer.

    Raises:
        ValueError: If a count is less than 1, or if `base_channels` is not a multiple of
            `groups`.
    """

    def __init__(self, in_channels: int, base_channels: int, depth: int, groups: int) -> None:
        super().__init__()
        if min(in_channels, base_channels, depth, groups) < 1:
            raise ValueError("in_channels, base_channels, depth, and groups must be 1 or more.")
        if base_channels % groups:
            raise ValueError(
                f"base_channels ({base_channels}) must be a multiple of groups ({groups})."
            )
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth
        self.groups = groups
        widths = [base_channels * 2**level for level in range(depth + 1)]
        self.encoder = nn.ModuleList()
        previous = in_channels
        for width in widths[:-1]:
            self.encoder.append(_Block(previous, width, groups))
            previous = width
        self.bottleneck = _Block(widths[-2], widths[-1], groups)
        self.decoder = nn.ModuleList(
            _Block(widths[level + 1] + widths[level], widths[level], groups)
            for level in reversed(range(depth))
        )
        self.output = nn.Conv2d(base_channels, 1, kernel_size=1)
        # Zero output at the start: a new model gives the mean path gain of the pretraining
        # inputs.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Calculate the output map of each image.

        Args:
            features: Input with shape `[batch, in_channels, rows, columns]`. `rows` and
                `columns` must be multiples of `2 ** depth`.

        Returns:
            The output with shape `[batch, rows, columns]`.

        Raises:
            ValueError: If `rows` or `columns` is not a multiple of `2 ** depth`.
        """
        multiple = 2**self.depth
        if features.shape[-2] % multiple or features.shape[-1] % multiple:
            raise ValueError(
                f"The rows and columns must be multiples of {multiple}, but the shape is "
                f"{tuple(features.shape)}."
            )
        skips = []
        x = features
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = functional.max_pool2d(x, kernel_size=2)
        x = self.bottleneck(x)
        for block, skip in zip(self.decoder, reversed(skips), strict=True):
            x = functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            x = block(torch.cat([x, skip], dim=1))
        return self.output(x)[:, 0]
