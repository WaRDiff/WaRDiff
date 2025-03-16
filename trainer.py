import os
import argparse

import matplotlib.pyplot as plt

from fvcore.nn import FlopCountAnalysis

import torch

import torch.optim as optim

from torch.utils.data import DataLoader

from wahfem import WaHFEM
from unet import UNetRes
from diffusion import ResidualDiffusion
from ema import EMAHelper

from mayo_dataset import Mayo2016Dataset, Mayo2020Dataset
from measure import *


class Trainer(object):
    def __init__(self, args, config):
        super().__init__()

        self.args = args
        self.config = config
        self.device = config.device

        self.dataset = config.dataset

        self.batch_size = config.train.batch_size
        self.num_workers = config.train.num_workers
        self.epochs = config.train.epochs

        if self.dataset == 'mayo2016':
            self.model_savepath = config.savepath2016.model_savepath
            self.validation_savepath = config.savepath2016.validation_savepath
            self.test_savepath = config.savepath2016.test_savepath

        elif self.dataset == 'mayo2020':
            self.model_savepath = config.savepath2020.model_savepath
            self.validation_savepath = config.savepath2020.validation_savepath
            self.test_savepath = config.savepath2020.test_savepath

        self.lr = config.optim.learning_rate
        self.weight_decay = config.optim.weight_decay

        self.ema_helper = EMAHelper()

        self.set_folder()
        self.log_config(log_type='train')

        self.wahfem = WaHFEM()
        self.unetres = UNetRes(self.config.model)

        self.residualdiffusion = ResidualDiffusion(self.wahfem, self.unetres, self.config).to(self.device)

        print("Self.device: ", self.device)
        print("Model device: ", next(self.residualdiffusion.parameters()).device)

        self.ema_helper.register(self.residualdiffusion)
        self.optimizer, self.scheduler = self.get_optimizer(self.config, self.residualdiffusion.parameters())

    
    def train(self):
        if self.dataset == 'mayo2016':
            train_dataset = Mayo2016Dataset('train', self.config, augmentation=self.config.train.augmentation)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

            validation_dataset = Mayo2016Dataset('validation', self.config, augmentation=False)
            validation_loader = DataLoader(validation_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False)

        elif self.dataset == 'mayo2020':
            train_dataset = Mayo2020Dataset('train', self.config, augmentation=self.config.train.augmentation)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

            validation_dataset = Mayo2020Dataset('validation', self.config, augmentation=False)
            validation_loader = DataLoader(validation_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False)

        clean_sample, noisy_sample = next(iter(train_loader))
        clean_sample = clean_sample.to(self.device)
        noisy_sample = noisy_sample.to(self.device)
        
        self.start_epoch = 1
        best_psnr = 0.0
        best_average = 0.0
        total_iters = 0

        if self.args.resume:
            self.load_model(self.model_savepath, load_type='latest')

        for epoch in range(self.start_epoch, self.epochs+1):
            print(f'Epoch {epoch} start!')
            self.residualdiffusion.train()
            self.residualdiffusion.unetres.train()
            self.residualdiffusion.wahfem.train()
            
            train_loss = 0.0

            num_iters = 0
            for iter_, (clean, noisy)in enumerate(train_loader):
                total_iters += 1
                num_iters += 1

                self.optimizer.zero_grad()

                clean = clean.to(self.device)
                noisy = noisy.to(self.device)

                diffusion_output = self.residualdiffusion(clean, noisy)
                loss = diffusion_output['loss']
                
                loss.backward()
                self.optimizer.step()

                train_loss += loss

            avg_train_loss = train_loss / num_iters
            print(f'Epoch {epoch} - Train Loss: {avg_train_loss}')

            self.scheduler.step()

            num_iters = 0
            val_psnr, val_ssim, val_lpips, val_vif = 0.0, 0.0, 0.0, 0.0

            self.residualdiffusion.eval()
            self.residualdiffusion.unetres.eval()
            self.residualdiffusion.wahfem.eval()

            with torch.no_grad():
                for val_iter_, (val_clean, val_noisy) in enumerate(validation_loader):
                    num_iters += 1
                    
                    val_clean = val_clean.to(self.device)
                    val_noisy = val_noisy.to(self.device)

                    val_pred, val_pred_ll = self.residualdiffusion.sample(val_noisy, val_clean)

                    val_pred_normed = self.normalize_zero_to_one(val_pred)
                    val_clean_normed = self.normalize_zero_to_one(val_clean)
                    val_noisy_normed = self.normalize_zero_to_one(val_noisy)

                    val_pred_metrices = self.compute_measure(val_pred_normed, val_clean_normed)
                    val_noisy_metrices = self.compute_measure(val_noisy_normed, val_clean_normed)

                    val_psnr += val_pred_metrices['psnr']
                    val_ssim += val_pred_metrices['ssim']
                    val_lpips + val_pred_metrices['lpips']
                    val_vif += val_pred_metrices['vif']
                    
                self.save_figure(val_noisy_normed, val_pred_normed, val_clean_normed, val_noisy_metrices, val_pred_metrices, 'validation', epoch, self.validation_savepath)

                avg_val_psnr = val_psnr / num_iters
                avg_val_ssim = val_ssim / num_iters
                avg_val_lpips = val_lpips / num_iters
                avg_val_vif = val_vif / num_iters

            self.save_model(epoch, 'latest')
            if avg_val_psnr > best_psnr:
                self.save_model(epoch, 'best_psnr')
                best_psnr = avg_val_psnr
                print('New Best Model Saved!')
            if best_average < ((avg_val_psnr + avg_val_ssim) / 2):
                self.save_model(epoch, 'best_average')
                best_average = ((avg_val_psnr + avg_val_ssim) / 2)
                print('New Best Average Model Saved!')

            print(f'EPOCH: {epoch} | PSNR: {avg_val_psnr} | SSIM: {avg_val_ssim} | LPIPS: {avg_val_lpips} | VIF: {avg_val_vif}')


    def test(self):
        self.log_config(log_type='test')

        test_dataset = Mayo2016Dataset('test', self.config, augmentation=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

        self.load_model(self.model_savepath, load_type='best_psnr', ema=True)

        input_psnr, input_ssim, input_lpips, input_vif = 0.0, 0.0, 0.0, 0.0
        test_psnr, test_ssim, test_lpips, test_vif = 0.0, 0.0, 0.0, 0.0

        self.residualdiffusion.eval()
        
        with torch.no_grad():
            for iter_, (clean, noisy) in enumerate(test_loader):
                clean = clean.to(self.device)
                noisy = noisy.to(self.device)

                pred = self.residualdiffusion.sample(noisy)

                pred_normed = self.normalize_zero_to_one(pred)
                clean_normed = self.normalize_zero_to_one(clean)
                noisy_normed = self.normalize_zero_to_one(noisy)

                pred_metrices = self.compute_measure(pred_normed, clean_normed)
                noisy_metrices = self.compute_measure(noisy_normed, clean_normed)

                input_psnr += noisy_metrices['psnr']
                input_ssim += noisy_metrices['ssim']
                input_lpips += noisy_metrices['lpips']
                input_vif += noisy_metrices['vif']

                test_psnr += pred_metrices['psnr']
                test_ssim += pred_metrices['ssim']
                test_lpips += pred_metrices['lpips']
                test_vif += pred_metrices['vif']

                self.save_figure(noisy_normed, pred_normed, clean_normed, noisy_metrices, pred_metrices, 'test', iter_, self.test_savepath)
                
            avg_input_psnr = input_psnr / len(test_dataset)
            avg_input_ssim = input_ssim / len(test_dataset)
            avg_input_lpips = input_lpips / len(test_dataset)
            avg_input_vif = input_vif / len(test_dataset)

            avg_test_psnr = test_psnr / len(test_dataset)
            avg_test_ssim = test_ssim / len(test_dataset)
            avg_test_lpips = test_lpips / len(test_dataset)
            avg_test_vif = test_vif / len(test_dataset)

            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nLPIPS avg: {:.4f} \nVIF avg: {:.4f}'.format(avg_input_psnr, avg_input_ssim, avg_input_lpips, avg_input_vif))
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nLPIPS avg: {:.4f} \nVIF avg: {:.4f}'.format(avg_test_psnr, avg_test_ssim, avg_test_lpips, avg_test_vif))
            print('n')

            with open(os.path.join(self.test_savepath, 'average_results.txt'), 'w') as avg_file:
                avg_file.write('Average Results:\n')
                avg_file.write('Original average results:\n')
                avg_file.write(f"PSNR: {avg_input_psnr}, SSIM: {avg_input_ssim}, LPIPS: {avg_input_lpips}, VIF: {avg_input_vif}\n\n")
                avg_file.write('Predicted average results:\n')
                avg_file.write(f"PSNR: {avg_test_psnr}, SSIM: {avg_test_ssim}, LPIPS: {avg_test_lpips} VIF: {avg_test_vif}\n")


    def get_gpu_model_info(self, model, device, clean_sample, noisy_sample):
        
        total_params = sum(p.numel() for p in model.parameters())
        model_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
        flop_count = FlopCountAnalysis(model, (clean_sample, noisy_sample))

        print(f"Model Parameters: {total_params:,}")
        print(f"Model Weights on GPU: {model_memory:.2f} MB")
        print(f"Total FLOPs: {flop_count.total()}")
        
    
    def log_config(self, log_type="train"):

        log_file = os.path.join(self.model_savepath, f"{log_type}_config_log.txt")

        print("\n" + "=" * 50)
        print(f"[{log_type.upper()} CONFIGURATION]")
        print("=" * 50)

        print("\nArgs:")
        for arg, value in vars(self.args).items():
            print(f"  {arg}: {value}")

        print("\n🔧 Config:")
        for key, value in vars(self.config).items():
            if isinstance(value, argparse.Namespace):  # 네임스페이스 내부 값도 출력
                print(f"  {key}:")
                for sub_key, sub_value in vars(value).items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")

        print("=" * 50 + "\n")

        with open(log_file, "w") as f:
            f.write("=" * 50 + "\n")
            f.write(f"[{log_type.upper()} CONFIGURATION]\n")
            f.write("=" * 50 + "\n\n")

            f.write("Args:\n")
            for arg, value in vars(self.args).items():
                f.write(f"  {arg}: {value}\n")

            f.write("\nConfig:\n")
            for key, value in vars(self.config).items():
                if isinstance(value, argparse.Namespace):
                    f.write(f"  {key}:\n")
                    for sub_key, sub_value in vars(value).items():
                        f.write(f"    {sub_key}: {sub_value}\n")
                else:
                    f.write(f"  {key}: {value}\n")

            f.write("=" * 50 + "\n")

        print(f"[{log_type.upper()}] Configuration saved to {log_file}")


    def normalize_zero_to_one(self, image):
        normalized = (image + 1) / 2

        return normalized


    def normalize_min_max(self, image):
        min_value = torch.min(image)
        max_value = torch.max(image)

        return (image - min_value) / (max_value - min_value)


    def compute_measure(self, noisy, clean, data_range=1.0):
        results = {}
        
        results['psnr'] = compute_psnr(noisy, clean, data_range=data_range)
        results['ssim'] = compute_ssim(noisy, clean, data_range=data_range)
        results['lpips'] = compute_lpips(noisy, clean)
        results['vif'] = compute_vif(noisy, clean, data_range=data_range)
        
        return results


    def save_figure(self, noisy, pred, clean, noisy_metrices, pred_metrices, save_type, epoch, save_path):
        clean = clean.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        noisy = noisy.detach().cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(30, 10))

        axs[0].imshow(noisy[0].transpose((1, 2, 0)), cmap='gray')
        axs[0].set_title('LDCT', fontsize=30)
        axs[0].set_xlabel('PSNR: {:.4f}\nSSIM: {:.4f}\n'.format(noisy_metrices['psnr'], noisy_metrices['ssim']))

        axs[1].imshow(pred[0].transpose((1, 2, 0)), cmap='gray')
        axs[1].set_title('Predicted', fontsize=30)
        axs[1].set_xlabel('PSNR: {:.4f}\nSSIM: {:.4f}\n'.format(pred_metrices['psnr'], pred_metrices['ssim']))

        axs[2].imshow(clean[0].transpose((1, 2, 0)), cmap='gray')
        axs[2].set_title('NDCT', fontsize=30)

        fig.savefig(os.path.join(save_path, "{}_{}.png".format(save_type, epoch)))
        plt.close()


    def save_model(self, epoch, save_type='best_psnr'):

        save_path = os.path.join(self.model_savepath, f"{save_type}_model.ckpt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {'epoch': epoch,
                    'diffusion': self.residualdiffusion.state_dict(),
                    'wahfem' : self.residualdiffusion.wahfem.state_dict(),
                    'unetres': self.residualdiffusion.unetres.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'ema_helper': self.ema_helper.state_dict(),
                    'params': self.args,
                    'config': self.config}

        torch.save(checkpoint, save_path)


    def load_model(self, load_path, load_type='best_psnr', ema=True):
        file_name = os.path.join(load_path, f'{load_type}_model.ckpt')
        checkpoint =  torch.load(file_name)
        
        self.residualdiffusion.load_state_dict(checkpoint['diffusion'], strict=True)
        self.residualdiffusion.wahfem.load_state_dict(checkpoint['wahfem'], strict=True)
        self.residualdiffusion.unetres.load_state_dict(checkpoint['unetres'], strict=True)
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch= checkpoint.get('epoch', self.start_epoch)
        self.args = checkpoint.get('params', self.args)
        self.config = checkpoint.get('config', self.config)

        if ema:
            self.ema_helper.ema(self.residualdiffusion)

        print("=> loaded checkpoint {} step {}".format(load_path, self.start_epoch))


    def get_optimizer(self, config, parameters):
        if config.optim.optimizer == 'Adam':
            optimizer = optim.Adam(parameters, lr=config.optim.learning_rate, weight_decay=config.optim.weight_decay,
                                    betas=(0.9, 0.999), amsgrad=config.optim.amsgrad, eps=config.optim.eps)
        elif config.optim.optimizer == 'AdamW':
            optimizer = optim.AdamW(parameters, lr=config.optim.learning_rate, weight_decay=config.optim.weight_decay,
                                    betas=(0.9, 0.999), amsgrad=config.optim.amsgrad, eps=config.optim.eps)
        elif config.optim.optimizer == 'RMSProp':
            optimizer = optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
        elif config.optim.optimizer == 'SGD':
            optimizer = optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError(f'Optimizer {config.optim.optimizer} not understood.')
    
        if config.optim.scheduler == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.optim.step_size, gamma=config.optim.gamma, last_epoch=-1)
        elif config.optim.scheduler == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.optim.T_max)
        elif config.optim.scheduler == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=config.optim.gamma, patience=config.optim.patience)
        elif config.optim.scheduler == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.optim.gamma)
        elif config.optim.scheduler == "None":
            scheduler = None  # 스케줄러 사용 안 함
        else:
            raise NotImplementedError(f'Scheduler {config.optim.scheduler} not understood.')
        
        return optimizer, scheduler


    def set_folder(self):
        os.makedirs(self.model_savepath, exist_ok=True)
        os.makedirs(self.validation_savepath, exist_ok=True)
        os.makedirs(self.test_savepath, exist_ok=True)
