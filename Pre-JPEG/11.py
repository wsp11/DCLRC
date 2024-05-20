import argparse

from torchvision.transforms.functional import to_pil_image
import os
import torchjpeg.codec
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import torch
from typing import Tuple
from typing import Optional
import torch.nn.functional as F

import torch
from torch import Tensor

#from ._codec_ops import *  # pylint: disable=import-error

__all__ = ["read_coefficients", "write_coefficients", "quantize_at_quality", "pixels_for_channel", "reconstruct_full_image"]  # pylint: disable=undefined-all-variable

def to_rgb(x: Tensor, data_range: float = 255) -> Tensor:
    assert data_range in [1.0, 255]

    # fmt: off
    rgb_from_ycbcr = torch.tensor([
        1, 0, 1.40200,
        1, -0.344136286, -0.714136286,
        1, 1.77200, 0
    ],
    device=x.device).view(3, 3).transpose(0, 1)
    # fmt: on

    if data_range == 255:
        b = torch.tensor([-179.456, 135.458816, -226.816], device=x.device).view(3, 1, 1)
    else:
        b = torch.tensor([-0.70374902, 0.531211043, -0.88947451], device=x.device).view(3, 1, 1)

    x = torch.einsum("cv,...cxy->...vxy", [rgb_from_ycbcr, x])
    x += b

    return x.contiguous()

def deblockify(blocks: Tensor, size: Tuple[int, int]) -> Tensor:
    bs = blocks.shape[0]
    ch = blocks.shape[1]
    block_size = blocks.shape[3]

    blocks = blocks.view(bs * ch, -1, int(block_size**2))
    #blocks = blocks.transpose(1, 2)
    blocks = blocks.permute(0, 2, 1)
    blocks = torch.nn.functional.fold(blocks, output_size=size, kernel_size=(block_size, block_size), stride=(block_size, block_size))
    blocks = blocks.view(bs, ch, size[0], size[1])

    return blocks

def idct(input: Tensor) -> Tensor:
    # Get dtype and device
    dtype: torch.dtype = input.dtype
    device: torch.device = input.device
    # Make and apply scaling
    alpha: Tensor = torch.ones(8, dtype=dtype, device=device)
    alpha[0] = 1.0 / (2**0.5)
    dct_scale: Tensor = torch.outer(alpha, alpha)
    input = input * dct_scale[None, None]
    # Make DCT tensor and scaling
    index: Tensor = torch.arange(8, dtype=dtype, device=device)
    x, y, u, v = torch.meshgrid(index, index, index, index)  # type: Tensor, Tensor, Tensor, Tensor
    idct_tensor: Tensor = ((2.0 * u + 1.0) * x * torch.pi / 16.0).cos() * ((2.0 * v + 1.0) * y * torch.pi / 16.0).cos()
    # Apply DCT
    output: Tensor = 0.25 * torch.tensordot(input, idct_tensor, dims=2) + 128.0
    return output
        
def pixels_for_channel(channel: Tensor, quantization: Tensor, crop: Optional[Tensor] = None) -> Tensor:
    dequantized = channel.float() * quantization.float()
    s = idct(dequantized)
    s = s.view(1, 1, s.shape[1] * s.shape[2], 8, 8)
    s = deblockify(s, (channel.shape[1] * 8, channel.shape[2] * 8))
    s = s.squeeze()

    return s

def chroma_subsampling(self,image):
    pool = nn.AvgPool2d(kernel_size=2, stride=2)
    out = pool(image)
    return out

def chroma_upsampling(y, cb, cr):
    """ Upsample chroma layers
    Input: 
        y(tensor): y channel image with shape (height, width)
        cb(tensor): cb channel with shape (height/2, width/2)
        cr(tensor): cr channel with shape (height/2, width/2)
    Output:
        image(tensor): image with shape (channels, height, width)
    """
    def repeat(x, k=2):
        height, width = x.shape
        x = x.view(-1, 1, height, 1, width, 1)
        x = x.repeat(1, k, 1, k, 1, 1)
        x = x.view(height * k, width * k)
        return x

    # Upsample cb and cr to match the size of y
    cb_upsampled = repeat(cb)
    cr_upsampled = repeat(cr)

    # Stack the channels to form the output image
    image = torch.stack([y, cb_upsampled, cr_upsampled], dim=0)
    
    return image

def reconstruct_full_image(y_coefficients: Tensor, quantization: Tensor, cbcr_coefficients: Optional[Tensor] = None, crop: Optional[Tensor] = None) -> Tensor:
    y = pixels_for_channel(y_coefficients, quantization[0])

    if cbcr_coefficients is not None:
        cb = pixels_for_channel(cbcr_coefficients[0:1], quantization[1])
        cr = pixels_for_channel(cbcr_coefficients[1:2], quantization[2])
        out = chroma_upsampling(y,cb,cr)
        out = to_rgb(out).squeeze()
    else:
        out = y

    return out.clamp(0, 255)/255.0

path = "/home/whut1/wsp/torchjpeg-master/examples/sample.jpg"
dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(path)
#print(Y_coefficients.shape)
spatial = torchjpeg.codec.reconstruct_full_image(Y_coefficients, quantization, CbCr_coefficients, dimensions)

dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(path)
spatial2 = reconstruct_full_image(Y_coefficients, quantization, CbCr_coefficients, dimensions)

mse = torch.nn.MSELoss(reduction='mean')
print(spatial)
print(spatial2)
print(spatial==spatial2)
mse = F.mse_loss(spatial, spatial2)
max_pixel = 1.0
psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
print(psnr.item())