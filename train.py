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
from model import SRDHnet
from data import HazeDataset
from image_quality_assessment import PSNR, SSIM
import pytorch_ssim


# @logger
def load_data(cfg):
    data_transform = transforms.Compose([
        transforms.Resize([480, 640]),
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
    SRDH = SRDHnet(upscale_factor=upscale_factor).to(device)
    SRDH.apply(weight_init)
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
    # -------------------------------------------------------------------
    # load network
    network = load_network(cfg.upscale_factor, device)
    # -------------------------------------------------------------------
    # load optimizer
    optimizer = load_optimizer(network, cfg)
    # -------------------------------------------------------------------
    # start train
    print('Start train')
    network.train()
    for epoch in range(cfg.epochs):
        trainloader = tqdm(train_loader)
        for step, (ori_image, haze_image, haze_img_lr) in enumerate(trainloader):
            count = epoch * train_number + (step + 1)
            ori_image, haze_image, haze_img_lr = ori_image.to(device), haze_image.to(device), haze_img_lr.to(device)
            dehaze_image = network(haze_img_lr)
            loss = criterion(dehaze_image, ori_image)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            summary.add_scalar('loss', loss.item(), count)
            if step % cfg.print_gap == 0:
                summary.add_image('DeHaze_Images', make_grid(dehaze_image[:4].data, normalize=True, scale_each=True),
                                  count)
                summary.add_image('Haze_Images', make_grid(haze_image[:4].data, normalize=True, scale_each=True), count)
                summary.add_image('Origin_Images', make_grid(ori_image[:4].data, normalize=True, scale_each=True),
                                  count)
            trainloader.set_description_str('Epoch: {}/{}  |  Step: {}/{}  '.format(epoch + 1, cfg.epochs, step + 1, train_number))
            trainloader.set_postfix_str('lr: {:.2f}  | Loss: {:.2f}'.format(optimizer.param_groups[0]['lr'], loss.item()))
            # print('Epoch: {}/{}  |  Step: {}/{}  |  lr: {:.6f}  | Loss: {:.6f}'
            #       .format(epoch + 1, cfg.epochs, step + 1, train_number,
            #               optimizer.param_groups[0]['lr'], loss.item()))
        # -------------------------------------------------------------------
        # start validation
        print('Epoch: {}/{} | Validation Model Saving Images'.format(epoch + 1, cfg.epochs))
        network.eval()
        with torch.no_grad():
            valloader = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            for step, (ori_image, haze_image, haze_img_lr) in enumerate(valloader):
                ori_image, haze_image, haze_img_lr = ori_image.to(device), haze_image.to(device), haze_img_lr.to(device)
                dehaze_image = network(haze_img_lr)
                valing_results['batch_sizes'] += cfg.batch_size
                if not step > 10:   # only save image 10 times
                    torchvision.utils.save_image(
                    torchvision.utils.make_grid(torch.cat((haze_image, dehaze_image, ori_image), 0),
                                                nrow=ori_image.shape[0]),
                    os.path.join(cfg.sample_output_folder, '{}_{}.jpg'.format(epoch + 1, step)))
                batch_mse = ((dehaze_image - ori_image) ** 2).data.mean()
                valing_results['mse'] += batch_mse * cfg.batch_size
                batch_ssim = pytorch_ssim.ssim(dehaze_image, ori_image).item()
                valing_results['ssims'] += batch_ssim * cfg.batch_size
                valing_results['psnr'] = 10 * log10((ori_image.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                valloader.set_postfix_str(
                    '[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
                
        
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
