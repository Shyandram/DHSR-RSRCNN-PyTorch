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
from model import GridDehazeNet
from data import HazeDataset
from image_quality_assessment import PSNR, SSIM
from loss import ContentLoss
import pytorch_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
from thop import profile

import warnings
warnings.filterwarnings("ignore")


# @logger
def load_data(cfg):
    train_data_transform = transforms.Compose([
        # transforms.Resize([480, 640]),
        # transforms.RandomCrop([256, 256]),
        transforms.CenterCrop([240, 240]),
        transforms.ToTensor()
    ])
    test_data_transform = transforms.Compose([
        transforms.Resize([480, 640]),
        # transforms.RandomCrop([256, 256]),
        transforms.CenterCrop([480, 640]),
        transforms.ToTensor()
    ])
    train_haze_dataset = HazeDataset(cfg.ori_data_path, cfg.haze_data_path, train_data_transform, cfg.upscale_factor)
    train_loader = torch.utils.data.DataLoader(train_haze_dataset, batch_size=cfg.batch_size, shuffle=True,
                                               num_workers=cfg.num_workers, drop_last=True, pin_memory=True)

    val_haze_dataset = HazeDataset(cfg.val_ori_data_path, cfg.val_haze_data_path, test_data_transform, cfg.upscale_factor, 
                                   issots=True)
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
    aod = GridDehazeNet().to(device)
    aod.apply(weight_init)
    return aod

def load_pretrain_network(cfg, device):
    aod = GridDehazeNet().to(device)
    aod.load_state_dict(torch.load(os.path.join(cfg.model_dir, cfg.net_name, cfg.ckpt))['state_dict'])
    return aod

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
    vggloss = ContentLoss(device=device)
    # -------------------------------------------------------------------
    # load network
    if cfg.ckpt:
        network = load_pretrain_network(cfg, device)
    else:
        network = load_network(cfg.upscale_factor, device) 
    flops, params = profile(network, inputs=(torch.zeros((1, 3, 480, 640), device=device), ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    # print('Total params: ', sum(p.numel() for p in network.parameters() if p.requires_grad))
    # -------------------------------------------------------------------
    # load optimizer
    optimizer = load_optimizer(network, cfg)
    # -------------------------------------------------------------------
    ssim = SSIM(data_range=1.).to(device=device)
    psnr = PSNR(data_range=1.).to(device=device)
    lpips = LPIPS().to(device=device)
    # start train
    print('Start train')
    network.train()
    for epoch in range(cfg.epochs):
        trainloader = tqdm(train_loader)
        for step, (ori_image, haze_image, haze_img_lr, ori_image_lr) in enumerate(trainloader):
            count = epoch * train_number + (step + 1)
            ori_image, haze_image, haze_img_lr, ori_image_lr = ori_image.to(device), haze_image.to(device), haze_img_lr.to(device), ori_image_lr.to(device)
            sr_image = network(haze_image)
            sr_mseloss = criterion(sr_image, ori_image)
            # # contloss = vggloss(sr_image, ori_image)
            # loss = sr_mseloss #+ 0.006  * contloss
                        
            sr_smooth_loss = torch.nn.functional.smooth_l1_loss(sr_image, ori_image)
            # perceptual_loss = vggloss(sr_image, ori_image)
            loss = sr_smooth_loss#+ 0.04*perceptual_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            summary.add_scalar('loss', loss.item(), count)
            summary.add_scalar('sr_mseloss', sr_mseloss.item(), count)
            # summary.add_scalar('contentloss', contloss.item(), count)
            trainloader.set_description_str('Epoch: {}/{}  |  Step: {}/{}  '.format(epoch + 1, cfg.epochs, step + 1, train_number))
            trainloader.set_postfix_str('lr: {:.2f}  | MSELoss:  {:.2f}'.format(optimizer.param_groups[0]['lr'], sr_mseloss.item()))
            # print('Epoch: {}/{}  |  Step: {}/{}  |  lr: {:.6f}  | Loss: {:.6f}'
            #       .format(epoch + 1, cfg.epochs, step + 1, train_number,
            #               optimizer.param_groups[0]['lr'], loss.item()))
        # -------------------------------------------------------------------
        # start validation
        print('Epoch: {}/{} | Validation Model Saving Images'.format(epoch + 1, cfg.epochs))
        network.eval()
        with torch.no_grad():
            valloader = tqdm(val_loader)
            valloader.set_description_str('DH & SR valloader')
            DH_valing_results = {'mse': 0, 'ssims': 0, 'psnrs': 0, 'lpipss': 0,'psnr': 0, 'ssim': 0, 'lpips': 0, 'batch_sizes': 0}
            SR_valing_results = {'mse': 0, 'ssims': 0, 'psnrs': 0, 'lpipss': 0,'psnr': 0, 'ssim': 0, 'lpips': 0, 'batch_sizes': 0}
            for step, (ori_image, haze_image, haze_img_lr, ori_image_lr) in enumerate(valloader):
                ori_image, haze_image, haze_img_lr, ori_image_lr = ori_image.to(device), haze_image.to(device), haze_img_lr.to(device), ori_image_lr.to(device)
                sr_image = network(haze_image)
                if not step > 2:   # only save image 10 times
                    torchvision.utils.save_image(
                    torchvision.utils.make_grid(torch.cat((haze_image, sr_image, ori_image), 0),
                                                nrow=ori_image.shape[0]),
                    os.path.join(cfg.sample_output_folder, '{}_{}.jpg'.format(epoch + 1, step)))
                # DH
                # SR
                SR_valing_results['batch_sizes'] += cfg.batch_size

                batch_ssim = ssim(sr_image, ori_image).item()
                SR_valing_results['ssims'] += batch_ssim * cfg.batch_size
                SR_valing_results['ssim'] = SR_valing_results['ssims'] / SR_valing_results['batch_sizes']

                batch_psnr = psnr(sr_image, ori_image).item()
                SR_valing_results['psnrs'] += batch_psnr * cfg.batch_size
                SR_valing_results['psnr'] = SR_valing_results['psnrs'] / SR_valing_results['batch_sizes']
                
                sr_image[sr_image>1.] = 1.
                sr_image[sr_image<0.] = 0.
                batch_lpips = lpips(sr_image, ori_image).item()
                SR_valing_results['lpipss'] += batch_lpips * cfg.batch_size
                SR_valing_results['lpips'] = SR_valing_results['lpipss'] / SR_valing_results['batch_sizes']

                summary.add_scalar('SR/PSNR', SR_valing_results['psnr'], epoch)
                summary.add_scalar('SR/ssim', SR_valing_results['ssim'], epoch)
                summary.add_scalar('SR/lpips', SR_valing_results['lpips'], epoch)

                # valloader.set_postfix_str(
                #     '[HZ to CLR] PSNR: %.4f dB SSIM: %.4f' % (
                #         DH_valing_results['psnr'], DH_valing_results['ssim']))
                valloader.set_postfix_str(
                    '[DH] PSNR: %.4f dB SSIM: %.4f LPIPS: %.4f' % (
                        SR_valing_results['psnr'], SR_valing_results['ssim'], SR_valing_results['lpips'],
                        ))
        network.train()
        # -------------------------------------------------------------------
        # save per epochs model
        save_model(epoch, cfg.model_dir, network, optimizer, cfg.net_name)
        # -------------------------------------------------------------------
        # train finish
        summary.close()


if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)
