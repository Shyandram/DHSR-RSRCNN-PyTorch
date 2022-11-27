import os
import torch
import torch.backends.cudnn
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from torch.cuda import amp
from math import log10
from torchvision import transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import weight_init #,logger
from config import get_config
from model import DHSRnet
from data import HazeDataset
from image_quality_assessment import PSNR, SSIM
from loss import ContentLoss
import pytorch_ssim

import warnings
warnings.filterwarnings("ignore")


# @logger
def load_data(cfg):
    data_transform = transforms.Compose([
        transforms.Resize([480, 640]),
        # transforms.RandomCrop([256, 256]),
        transforms.CenterCrop([480, 480]),
        transforms.ToTensor()
    ])
    train_haze_dataset = HazeDataset(cfg.ori_data_path, cfg.haze_data_path, data_transform, cfg.upscale_factor)
    train_loader = torch.utils.data.DataLoader(train_haze_dataset, batch_size=cfg.batch_size, shuffle=True,
                                               num_workers=cfg.num_workers, drop_last=True, pin_memory=True)

    val_haze_dataset = HazeDataset(cfg.val_ori_data_path, cfg.val_haze_data_path, data_transform, cfg.upscale_factor)
    val_loader = torch.utils.data.DataLoader(val_haze_dataset, batch_size=cfg.val_batch_size, shuffle=False,
                                             num_workers=cfg.num_workers, drop_last=True, pin_memory=True)

    return train_loader, len(train_loader), val_loader, len(val_loader)


# @logger
def save_model(epoch, path, net, optimizer, net_name):
    if not os.path.exists(os.path.join(path, net_name)):
        os.makedirs(os.path.join(path, net_name))
    torch.save({'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
               f=os.path.join(path, net_name, '{}_{}.pkl'.format('SRDH', epoch)))


# @logger
def load_network(upscale_factor, device):
    SRDH = DHSRnet(upscale_factor=upscale_factor).to(device)
    SRDH.apply(weight_init)
    return SRDH

def load_pretrain_network(cfg, device):
    SRDH = DHSRnet().to(device)
    SRDH.load_state_dict(torch.load(os.path.join(cfg.model_dir, cfg.net_name, cfg.ckpt))['state_dict'])
    return SRDH

# @logger
def load_optimizer(net, cfg):
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optimizer


# @logger
def loss_func(device):
    criterion = torch.nn.MSELoss().to(device)
    return criterion


# @logger
def load_summaries(cfg):
    summary = SummaryWriter(log_dir=os.path.join(cfg.log_dir, cfg.net_name), comment='')
    return summary


def main(cfg):
    # -------------------------------------------------------------------
    # basic config
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------------------------
    # load summaries
    summary = load_summaries(cfg)
    # -------------------------------------------------------------------
    # load data
    train_loader, train_number, val_loader, val_number = load_data(cfg)
    if not os.path.isdir(cfg.sample_output_folder):
        os.makedirs(cfg.sample_output_folder)
    # -------------------------------------------------------------------
    # load loss
    criterion = loss_func(device)
    # vggloss = ContentLoss(device=device)
    # -------------------------------------------------------------------
    # load network
    if cfg.ckpt:
        network = load_pretrain_network(cfg, device)
    else:
        network = load_network(cfg.upscale_factor, device) 
    # print(sum(p.numel() for p in network.parameters() if p.requires_grad))
    # -------------------------------------------------------------------
    # start train
    print('Start eval')
    network.eval()
    with torch.no_grad():
        valloader = tqdm(val_loader)
        valloader.set_description_str('DH & SR valloader')
        DH_valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        SR_valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        for step, (ori_image, haze_image, haze_img_lr, ori_image_lr) in enumerate(valloader):
            ori_image, haze_image, haze_img_lr, ori_image_lr = ori_image.to(device), haze_image.to(device), haze_img_lr.to(device), ori_image_lr.to(device)
            dh_image, sr_image = network(haze_img_lr)
            if not step > 2:   # only save image 10 times
                torchvision.utils.save_image(
                torchvision.utils.make_grid(torch.cat((haze_image, sr_image, ori_image), 0),
                                            nrow=ori_image.shape[0]),
                os.path.join(cfg.sample_output_folder, '{}_{}.jpg'.format(epoch + 1, step)))
            # DH
            DH_valing_results['batch_sizes'] += cfg.batch_size
            batch_mse = ((dh_image - ori_image_lr) ** 2).data.mean()
            DH_valing_results['mse'] += batch_mse * cfg.batch_size
            batch_ssim = pytorch_ssim.ssim(dh_image, ori_image_lr).item()
            DH_valing_results['ssims'] += batch_ssim * cfg.batch_size
            DH_valing_results['psnr'] = 10 * log10((ori_image_lr.max()**2) / (DH_valing_results['mse'] / DH_valing_results['batch_sizes']))
            DH_valing_results['ssim'] = DH_valing_results['ssims'] / DH_valing_results['batch_sizes']
            
            summary.add_scalar('DH/PSNR', DH_valing_results['psnr'], count)
            summary.add_scalar('DH/ssim', DH_valing_results['ssim'], count)
            # SR
            SR_valing_results['batch_sizes'] += cfg.batch_size
            batch_mse = ((sr_image - ori_image) ** 2).data.mean()
            SR_valing_results['mse'] += batch_mse * cfg.batch_size
            batch_ssim = pytorch_ssim.ssim(sr_image, ori_image).item()
            SR_valing_results['ssims'] += batch_ssim * cfg.batch_size
            SR_valing_results['psnr'] = 10 * log10((ori_image.max()**2) / (SR_valing_results['mse'] / SR_valing_results['batch_sizes']))
            SR_valing_results['ssim'] = SR_valing_results['ssims'] / SR_valing_results['batch_sizes']

            summary.add_scalar('SR/PSNR', SR_valing_results['psnr'], count)
            summary.add_scalar('SR/ssim', SR_valing_results['ssim'], count)

            # valloader.set_postfix_str(
            #     '[HZ to CLR] PSNR: %.4f dB SSIM: %.4f' % (
            #         DH_valing_results['psnr'], DH_valing_results['ssim']))
            valloader.set_postfix_str(
                '[HZ to CLR] PSNR: %.4f dB SSIM: %.4f;[LR to SR] PSNR: %.4f dB SSIM: %.4f' % (
                    DH_valing_results['psnr'], DH_valing_results['ssim'], SR_valing_results['psnr'], SR_valing_results['ssim']))
        
        # train finish
        summary.close()


if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)
