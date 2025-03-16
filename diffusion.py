from collections import namedtuple
from einops import reduce
from functools import partial

import torch

import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ms_ssim

from dwt_idwt import dwt, idwt


ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    
    
def histogram_loss(pred, target, bins=256):
    pred_hist = torch.histc(pred, bins, min=target.min(), max=target.max())
    target_hist = torch.histc(target, bins, min=target.min(), max=target.max())
    return torch.mean(torch.abs(pred_hist - target_hist))


def unnormalize_to_zero_to_one(image):
        min_value = torch.min(image)
        max_value = torch.max(image)

        return (image - min_value) / (max_value - min_value)


def normalize_dwt(LL, HF):
    LL_norm = 2 * (LL - LL.min()) / (LL.max() - LL.min()) - 1
    HF_norm = 2 * (HF - HF.min()) / (HF.max() - HF.min()) - 1
    
    return LL_norm, HF_norm


def denormalize_dwt(pred_LL, pred_HF, original_LL, original_HF):
    pred_LL = (pred_LL + 1) / 2 * (original_LL.max() - original_LL.min()) + original_LL.min()
    pred_HF = (pred_HF + 1) / 2 * (original_HF.max() - original_HF.min()) + original_HF.min()
    
    return pred_LL, pred_HF


def scaling(clean, noisy):
    clean_min, clean_max = torch.min(clean), torch.max(clean)
    noisy_min, noisy_max = torch.min(noisy), torch.max(noisy)
    
    min_max_normed = (noisy - noisy_min) / (noisy_max - noisy_min)
    scaled = min_max_normed * (clean_max - clean_min) + clean_min
    
    return scaled


def gen_coefficients(timesteps, schedule="increased", sum_scale=1):
    if schedule == "increased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
        scale = 0.5*timesteps*(timesteps+1)
        alphas = x/scale
    elif schedule == "decreased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
        x = torch.flip(x, dims=[0])
        scale = 0.5*timesteps*(timesteps+1)
        alphas = x/scale
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float64)
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float64)
    assert alphas.sum()-torch.tensor(1) < torch.tensor(1e-10)

    return alphas*sum_scale


def identity(t, *args, **kwargs):
    return t


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class ResidualDiffusion(nn.Module):
    def __init__(self, wahfem, unetres, config):
        super().__init__()

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        self.wahfem = wahfem
        self.unetres = unetres

        self.image_size = config.diffusion.image_size
        self.timesteps = config.diffusion.timesteps
        self.sampling_timesteps = config.diffusion.sampling_timesteps
        self.objective = config.diffusion.objective
        self.use_wahfem = config.diffusion.use_wahfem
        self.ddim_sampling_eta = config.diffusion.ddim_sampling_eta
        self.loss_type = config.diffusion.loss_type
        self.wavelet_type = config.diffusion.wavelet_type
        self.condition = config.diffusion.condition

        self.noise_coff = config.diffusion.noise_coff
        self.freq_coff = config.diffusion.freq_coff
        self.photo_coff = config.diffusion.photo_coff

        self.output = {}
        
        self.noise_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.TVLoss = TVLoss()
        self.content_loss = nn.L1Loss()

        if self.objective == 'pred_res_noise':
            alphas = gen_coefficients(self.timesteps, schedule="decreased")
            alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2 = gen_coefficients(self.timesteps, schedule="increased")
            betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)
            betas_cumsum = torch.sqrt(betas2_cumsum)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
            posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
            posterior_variance[0] = 0

            timesteps, = alphas.shape
            self.num_timesteps = int(timesteps)

            assert self.sampling_timesteps <= timesteps
            self.is_ddim_sampling = self.sampling_timesteps < timesteps
            self.ddim_sampling_eta = self.ddim_sampling_eta

            register_buffer('alphas', alphas)
            register_buffer('alphas_cumsum', alphas_cumsum)
            register_buffer('one_minus_alphas_cumsum', 1-alphas_cumsum)
            register_buffer('betas2', betas2)
            register_buffer('betas', torch.sqrt(betas2))
            register_buffer('betas2_cumsum', betas2_cumsum)
            register_buffer('betas_cumsum', betas_cumsum)
            register_buffer('posterior_mean_coef1',
                            betas2_cumsum_prev/betas2_cumsum)
            register_buffer('posterior_mean_coef2', (betas2 *
                            alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum)
            register_buffer('posterior_mean_coef3', betas2/betas2_cumsum)
            register_buffer('posterior_variance', posterior_variance)
            register_buffer('posterior_log_variance_clipped',
                            torch.log(posterior_variance.clamp(min=1e-20)))

            self.posterior_mean_coef1[0] = 0
            self.posterior_mean_coef2[0] = 0
            self.posterior_mean_coef3[0] = 1
            self.one_minus_alphas_cumsum[-1] = 1e-6
        
    
    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return (
            x_t-extract(self.alphas_cumsum, t, x_t.shape) * x_res -
            extract(self.betas_cumsum, t, x_t.shape) * noise
        )


    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )

        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):
        return (x_t-extract(self.alphas, t, x_t.shape) * x_res -
                (extract(self.betas2, t, x_t.shape)/extract(self.betas_cumsum, t, x_t.shape)) * noise)
    

    def model_predictions(self, x_t, t, x_condition, clip_denoised=True):
        model_output = self.unetres(x_t, t, x_condition)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_denoised else identity

        if self.objective == 'pred_res_noise':
            pred_res = model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)

            x_start = self.predict_start_from_res_noise(
                x_t, t, pred_res, pred_noise)
            x_start = maybe_clip(x_start)

        return ModelResPrediction(pred_res, pred_noise, x_start)


    def p_mean_variance(self, x_t, t, x_condition):
        preds = self.model_predictions(x_t, t, x_condition)

        pred_res = preds.pred_res
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x_start, x_t=x_t, t=t)
        
        return model_mean, posterior_variance, posterior_log_variance, x_start


    def p_sample(self, x_t, t, x_condition):
        b, *_, device = *x_t.shape, x_t.device

        batched_times = torch.full(
            (x_t.shape[0],), t, device=x_t.device, dtype=torch.long)

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x_t=x_t, t=batched_times, x_condition=x_condition)

        noise = torch.randn_like(x_t) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred_img, x_start


    def p_sample_loop(self, x_t, x_condition):
        img = x_t

        for t in reversed(range(0, self.sampling_timesteps)):
            img, x_start = self.p_sample(x_t=img, t=t, x_condition=x_condition)

        return img
    

    def sample(self, x_noisy, x_clean):
        clean_ll, clean_high = dwt(x_clean)
        noisy_ll, noisy_high = dwt(x_noisy)
        
        clean_ll_normed, clean_high_normed = normalize_dwt(clean_ll, clean_high)
        noisy_ll_normed, noisy_high_normed = normalize_dwt(noisy_ll, noisy_high)

        wahfem = self.wahfem(clean_high)

        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        if self.use_wahfem:
            pred_clean_ll = sample_fn(noisy_ll_normed, wahfem)
            pred_clean_ll_denormed, wahfem_denormed = denormalize_dwt(pred_clean_ll, wahfem, clean_ll, clean_high)
            pred_clean = torch.cat([pred_clean_ll_denormed, wahfem_denormed], dim=1)
        else:
            pred_clean_ll = sample_fn(noisy_ll, noisy_high)
            pred_clean_ll_denormed, noisy_high_denormed = denormalize_dwt(pred_clean_ll, noisy_high_normed, clean_ll, clean_high)
            pred_clean = torch.cat([pred_clean_ll, noisy_high_denormed], dim=1)
        
        pred_clean = idwt(pred_clean)
        pred_clean = 2 * (pred_clean - pred_clean.min()) / (pred_clean.max() - pred_clean.min()) - 1
        
        return pred_clean, pred_clean_ll
            

    def q_sample(self, x_start, x_res, t, noise):
        return (
            x_start + extract(self.alphas_cumsum, t, x_start.shape) * x_res +
            extract(self.betas_cumsum, t, x_start.shape) * noise
        )
    

    def ddim_sample(self, x_t, x_condition):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            x_t.shape[0], x_t.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
    
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        image_list = []
        img = x_t

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            preds = self.model_predictions(img, time_cond, x_condition)
            
            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                betas2_cumsum_next-sigma2).sqrt()/betas_cumsum
            
            noise = torch.randn_like(img)

            img = img - alpha*pred_res - \
                    (betas_cumsum-(betas2_cumsum_next-sigma2).sqrt()) * \
                    pred_noise + sigma2.sqrt()*noise

        return img

    
    def beta_scheduler(self, beta_schedule, beta_start, beta_end, timesteps):
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "sigmoid":
            betas = torch.linspace(-6, 6, timesteps)
            betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented")
        
        return betas
    

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')


    def p_losses(self, x_t, t, x_condition, noise=None):

        clean_ll, clean_high = dwt(x_t)
        noisy_ll, noisy_high = dwt(x_condition)

        x_res = noisy_ll - clean_ll
        noise = default(noise, lambda: torch.randn_like(x_res))

        x_t_noised = self.q_sample(clean_ll, x_res, t, noise)

        pred_wahfem = self.wahfem(noisy_high)
        
        model_output = self.unetres(x_t_noised, t, clean_high)

        target = []
        if self.objective == 'pred_res_noise':
            target.append(x_res) 
            target.append(noise)

            pred_res = model_output[0]
            pred_noise = model_output[1]

        else:
            raise ValueError(f'unknown objective {self.objective}')

        pred_x0, pred_ll = self.sample(x_condition, x_t)
        
        # ======= noise loss =======
        residual_loss, histo_loss, noise_msssim_loss = 0, 0, 0
        for i in range(len(model_output)):
            residual_loss = residual_loss + self.loss_fn(model_output[i], target[i], reduction='none')
            histo_loss = histo_loss + histogram_loss(model_output[i], target[i])
            noise_msssim_loss = 1 - ms_ssim((model_output[i]+1)/2, (target[i]+1)/2, data_range=1.0).to(x_t.device)
        
        noise_loss = residual_loss + 0.001*histo_loss + 0.23 * noise_msssim_loss
        
        # ======= frequency loss =======
        pred_wahfem_unnorm = unnormalize_to_zero_to_one(pred_wahfem)
        clean_high_unnorm = unnormalize_to_zero_to_one(clean_high)
        
        high_frequency_loss = self.l2_loss(pred_wahfem, clean_high)
        low_frequency_loss = self.l2_loss(pred_ll, clean_ll)
        tv_loss = self.TVLoss(pred_wahfem)
        msssim_loss = 1 - ms_ssim(pred_wahfem_unnorm, clean_high_unnorm, data_range=1.0).to(x_t.device)

        frequency_loss = 1.0*high_frequency_loss + 1.0*low_frequency_loss + 0.5*msssim_loss + 0.01*tv_loss

        # ======= photo loss =======
        pred_x0_unnorm = unnormalize_to_zero_to_one(pred_x0)
        clean_unnorm = unnormalize_to_zero_to_one(x_t)
        
        content_loss = self.l1_loss(pred_x0, x_t)
        msssim_loss = 1 - ms_ssim(pred_x0_unnorm, clean_unnorm, data_range=1.0).to(x_t.device)
        
        photo_loss = content_loss + msssim_loss
        
        # ======= total loss =======
        loss = self.noise_coff * noise_loss + self.freq_coff * frequency_loss + self.photo_coff * photo_loss
        loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()

        self.output['x_t_noised'] = x_t_noised
        self.output['pred_res'] = pred_res
        self.output['pred_noise'] = pred_noise
        self.output['true_res'] = x_res
        self.output['true_noise'] = noise
        self.output['pred_ll'] = pred_ll
        self.output['pred_wahfem'] = pred_wahfem
        self.output['clean_ll'] = clean_ll
        self.output['noisy_ll'] = noisy_ll
        self.output['clean_high'] = clean_high
        self.output['noisy_high'] = noisy_high
        self.output['pred_x0'] = pred_x0
        self.output['loss'] = loss
        

    def forward(self, x_t, x_condition):
        b, c, h, w = x_t.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=x_t.device).long()
        
        self.p_losses(x_t, t, x_condition)

        return self.output
