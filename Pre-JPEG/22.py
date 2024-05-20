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

import torch
from torch import Tensor

def jpeg_decode(
    input_y: Tensor,
    input_cb: Tensor,
    input_cr: Tensor,
    jpeg_quality: Tensor,
    H: int,
    W: int,
    floor_function: Callable[[Tensor], Tensor],
    clipping_function: Callable[[Tensor, int, int], Tensor],
    quantization_table_y: Tensor,
    quantization_table_c: Tensor,
) -> Tensor:
    assert isinstance(input_y, Tensor), f"Compressed Y component (input_y) must be a torch.Tensor, got {type(input_y)}."
    assert isinstance(
        input_cb, Tensor
    ), f"Compressed Cb component (input_cb) must be a torch.Tensor, got {type(input_cb)}."
    assert isinstance(
        input_cr, Tensor
    ), f"Compressed Cr component (input_cr) must be a torch.Tensor, got {type(input_cr)}."
    assert isinstance(
        jpeg_quality, Tensor
    ), f"Compression strength (jpeg_quality) must be a torch.Tensor, got {type(jpeg_quality)}."
    assert isinstance(H, int) and (H > 0), f"Height (H) must be as positive integer, got {H}."
    assert isinstance(W, int) and (W > 0), f"Width (W) must be as positive integer, got {H}."
    assert input_y.shape[0] == jpeg_quality.shape[0], (
        f"Batch size of Y components and compression strength must match, "
        f"got image shape {input_y.shape[0]} and compression strength shape {jpeg_quality.shape[0]}"
    )
    assert isinstance(
        quantization_table_y, Tensor
    ), f"QT (quantization_table_y) must be a torch.Tensor, got {type(quantization_table_y)}."
    assert isinstance(
        quantization_table_c, Tensor
    ), f"QT (quantization_table_c) must be a torch.Tensor, got {type(quantization_table_c)}."
    assert quantization_table_y.shape == (8, 8,), (
        f"QT (quantization_table_y) must have the shape [8, 8], " f"got {quantization_table_y.shape}"
    )
    assert quantization_table_c.shape == (8, 8,), (
        f"QT (quantization_table_c) must have the shape [8, 8], " f"got {quantization_table_c.shape}"
    )
    # QT to device
    quantization_table_y = quantization_table_y.to(input_y.device)
    quantization_table_c = quantization_table_c.to(input_cb.device)
    # Dequantize inputs
    input_y = dequantize(
        input_y,
        jpeg_quality,
        quantization_table_y,
        clipping_function=clipping_function,
        floor_function=floor_function,
    )
    input_cb = dequantize(
        input_cb,
        jpeg_quality,
        quantization_table_c,
        clipping_function=clipping_function,
        floor_function=floor_function,
    )
    input_cr = dequantize(
        input_cr,
        jpeg_quality,
        quantization_table_c,
        clipping_function=clipping_function,
        floor_function=floor_function,
    )
    # Perform inverse DCT
    idct_y: Tensor = idct_8x8(input_y)
    idct_cb: Tensor = idct_8x8(input_cb)
    idct_cr: Tensor = idct_8x8(input_cr)
    # Reverse patching
    image_y: Tensor = unpatchify_8x8(idct_y, H, W)
    image_cb: Tensor = unpatchify_8x8(idct_cb, H // 2, W // 2)
    image_cr: Tensor = unpatchify_8x8(idct_cr, H // 2, W // 2)
    # Perform chroma upsampling
    image_cb = chroma_upsampling(image_cb)
    image_cr = chroma_upsampling(image_cr)
    # Convert back into RGB space
    rgb_decoded: Tensor = ycbcr_to_rgb(torch.stack((image_y, image_cb, image_cr), dim=-1))
    rgb_decoded = rgb_decoded.permute(0, 3, 1, 2)
    return rgb_decoded

path = "/home/whut1/wsp/torchjpeg-master/examples/sample.jpg"
dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(path)
Y_coefficients = Y_coefficients.squeeze(0)
CbCr_coefficients = CbCr_coefficients.squeeze(0)
Cb_coefficients = CbCr_coefficients[0:1]
Cr_coefficients = CbCr_coefficients[1:2]
quantization_table_y = quantization[0:1]
quantization_table_c = quantization[1:2]
spatial = jpeg_decode(
        input_y=Y_coefficients,
        input_cb=Cb_coefficients,
        input_cr=Cb_coefficients,
        jpeg_quality=60,
        H=128,
        W=256,
        quantization_table_c=quantization_table_c,
        quantization_table_y=quantization_table_y,
        floor_function=floor_function,
        clipping_function=clipping_function,
    )

'''
dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(path)
spatial = torchjpeg.codec.reconstruct_full_image(Y_coefficients, quantization, CbCr_coefficients, dimensions)
'''
dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(path)
spatial2 = reconstruct_full_image(Y_coefficients, quantization, CbCr_coefficients, dimensions)


print(spatial)
print(spatial2)