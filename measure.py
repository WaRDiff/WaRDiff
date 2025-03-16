import lpips

from piq import vif_p

import torch

import torch.nn.functional as F

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def compute_psnr(noisy, clean, data_range=1.0):
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(noisy.device)

    return psnr_metric(noisy, clean)


def compute_ssim(noisy, clean, data_range=1.0):
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(noisy.device)

    return ssim_metric(noisy, clean)


def compute_lpips(noisy, clean, model='alex'):
    lpips_model = lpips.LPIPS(net=model, verbose=False).to(noisy.device)
    lpips_value = lpips_model(noisy, clean).mean().item()

    return lpips_value


def compute_vif(noisy, clean, data_range=1.0):
    noisy = torch.clamp(noisy, 0, 1)
    clean = torch.clamp(clean, 0, 1)

    return vif_p(noisy, clean, data_range=data_range)
