The PyTorch implementation of our paper:

**JPEG Image Lossy Recompression with Mutual Information Enhancement**

### :bookmark:Brief Introduction
Despite JPEG remaining the most prevalent image compression algorithm, the majority of these algorithms primarily concentrate on uncompressed images, thereby overlooking the substantial quantity of existing compressed JPEG images. The recent strides in JPEG recompression strive to further minimize these JPEG file sizes. Unfortunately, the prevalent techniques, especially lossy recompression, frequently fail to fully utilize the correlation of coefficients, resulting in less-than-optimal compression. Moreover, they do not tap into the similarities among JPEG images to boost compression. This paper proposes a novel approach, Distributed Coding for JPEG Lossy Recompression (DCLRC), to enhance the efficiency of JPEG recompression. DCLRC employs an encoding network to eliminate redundant data within the Discrete Cosine Transform (DCT) domain. This is followed by preprocessing via DCT coefficient reconstruction and alignment networks allocated for dimensionality reduction and channel alignment. More significantly, DCLRC capitalizes on the visually similar images in the DCT domain by incorporating a mutual information enhancement module. This pioneering module combines feature extraction, multi-head cross-attention, and information fusion. Experiments indicate that in comparison to existing JPEG lossy recompression techniques, DCLRC performs more competently, achieving an average PSNR improvement of 2.0 dB as well as an average MS-SSIM improvement of 2.3 dB on the KITTI dataset at low bitrates. Such results unequivocally verify the effectiveness of DCLRC in augmenting JPEG recompression performance. 

### Dataset
The datasets used for experiments are KITTI Stereo and Cityscape.

For KITTI Stereo you can download the necessary image pairs from [KITTI 2012](http://www.cvlibs.net/download.php?file=data_stereo_flow_multiview.zip) and [KITTI 2015](http://www.cvlibs.net/download.php?file=data_scene_flow_multiview.zip). After obtaining `data_stereo_flow_multiview.zip` and `data_scene_flow_multiview.zip`, run the following commands:
```bash
unzip data_stereo_flow_multiview.zip # KITTI 2012
mkdir data_stereo_flow_multiview
mv training data_stereo_flow_multiview
mv testing data_stereo_flow_multiview

unzip data_scene_flow_multiview.zip # KITTI 2015
mkdir data_scene_flow_multiview
mv training data_scene_flow_multiview
mv testing data_scene_flow_multiview
```

For Cityscape you can download the image pairs from [here](https://www.cityscapes-dataset.com/downloads/). After downloading `leftImg8bit_trainvaltest.zip` and `rightImg8bit_trainvaltest.zip`, run the following commands:
```bash
mkdir cityscape_dataset
unzip leftImg8bit_trainvaltest.zip
mv leftImg8bit cityscape_dataset
unzip rightImg8bit_trainvaltest.zip
mv rightImg8bit cityscape_dataset
```
