# WaRDiff: Wavelet based Residual Diffusion for CT Image Denoising (MICCAI 2025)

Eunji Kim, Hyojeong Lee, Sanghyun Park

## Abstract
Low-dose computed tomography (LDCT) reduces radiation exposure compared to normal-dose computed tomography (NDCT), but it suffers from high levels of noise that degrade image quality and hinder accurate diagnosis. When denoising LDCT, different types of noise, such as Gaussian or Poisson noise, must be considered while preserving fine details such as anatomical structures and edges. However, most diffusion-based denoising methods operate in the spatial domain and rely on Gaussian noise, often resulting in overly smooth outputs that fail to capture the diverse noise patterns in CT images. To mitigate this problem, we propose WaRDiff, a wavelet-based residual diffusion that uses both residual and noise to effectively remove CT noise. First, we analyze which wavelet subband has the greatest impact on CT image quality. Based on this analysis, we design a diffusion model in which the low-frequency component undergoes the forward diffusion process, and the high-frequency components are incorporated as conditioning information. To effectively remove noise from the high-frequency components before using them as conditions in the diffusion process, we design the wavelet high-frequency enhancement module (WaHFEM). Extensive experiments on the public LDCT datasets showed that our WaRDiff outperforms existing state-of-the-art methods. The source code is available at https://github.com/WaRDiff/WaRDiff.


## Datasets

The 2016 AAPM-Mayo dataset [link][https://ctcicblog.mayo.edu/2016-low-dose-ct-grand-challenge/]

The 2020 AAPM-Mayo dataset [link][https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/]


## Getting Started

### Train


### Test




