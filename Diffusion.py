from Modules import *
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

class Diffusion:
    def __init__(
            self,
            image_size: int,
            num_channels: int,
            num_timesteps: int,
            beta_min: float,
            beta_max: float,
            batch_size: int,
            beta_schedule: str = "quadratic",
            device: str = "cuda"
    ):
        """
        Initializes a Diffusion Model with the given parameters.

        Args:
            image_size (int): The size of the 2D input image (assumed to be square).
            num_channels (int): The number of channels in the input image. 
            num_timesteps (int): The number of timesteps to run the diffusion process for.
            beta_min (float): The minimum value of beta for the diffusion process.
            beta_max (float): The maximum value of beta for the diffusion process.
            batch_size (int): The batch size to use for training.
            beta_schedule (str): The schedule to use for beta. 
                The options are: 'quadratic', 'linear', 'cosine'. 
            device (str): The device to use for training. The options are: 'cuda', 'cpu'.

        Returns:
            None

        """
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.batch_size = batch_size
        self.device = device
        self.start_epoch = 1

        if beta_schedule == "quadratic":
            self.beta = torch.Tensor([beta_min + 
                                      (beta_max/num_timesteps**2)*x**2
                                        for x in range(num_timesteps)]).to(device)
        elif beta_schedule == "linear":
            self.beta = torch.linspace(beta_min, beta_max, num_timesteps).to(device)
        elif beta_schedule == "cosine":
            self.beta = cosine_beta_schedule(num_timesteps, beta_min, beta_max, s=0.008).to(device)
        else:
            raise ValueError("Invalid beta_schedule: {}".format(beta_schedule), 
                                ". The options are: 'quadratic', 'linear', 'cosine'.")
        self.alpha = torch.ones(num_timesteps).to(device) - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        return None
    
    
    def forward_process(
            self,
            images,
            times
    ):
        """
        Runs the forward diffusion process on the given images for the given times.

        Args:
            images (torch.Tensor): The images to run the forward diffusion process on.
            times (torch.Tensor): The times to run the forward diffusion process for.

        Returns:
            noisy_images (torch.Tensor), noise (torch.Tensor): 
                The noisy images after the forward process, and the noise that was added.
        """
        assert images.shape[0] == times.shape[0]
        assert images.shape[1] == self.num_channels
        assert images.shape[2] == self.image_size
        assert images.shape[3] == self.image_size

        images = images.to(self.device)
        times = times.to(self.device)

        noise = torch.randn_like(images).to(self.device)
        alpha_bar_t = self.alpha_bar[times].to(self.device)
        noisy_images = torch.sqrt(alpha_bar_t) * images + torch.sqrt(1 - alpha_bar_t) * noise

        return noisy_images, noise
    
    def create_model(
            self,
            num_init_ch=64,
            num_downsamples=3,
            num_mid_convs=3,
    ):
        """
        Initializes the U-net model.

        Args:
            num_init_ch (int): The number of channels in the first layer of the U-net.
            num_downsamples (int): The number of downsampling layers in the U-net.
            num_mid_convs (int): The number of convolutional layers in the middle of the U-net.

        Returns:
            model (nn.Module): The initialized model.
        """
        self.model = UNet(
            image_size=self.image_size,
            in_ch=self.num_channels,
            out_ch=self.num_channels,
            num_init_ch=num_init_ch,
            num_downsamples=num_downsamples,
            num_mid_convs=num_mid_convs,
            device=self.device
            )
        self.model = torch.compile(self.model)
        return self.model
    
    def load_from_checkpoint(
            self,
            checkpoint_path,
            model,
            optimizer=None,
            learning_scheduler=None,
            start_epoch=1
    ):
        """
        Loads a model from checkpoint_path.

        Args:
            checkpoint_path (str): The path to the model to load.
            model (nn.Module): The model to load the checkpoint into.
            optimizer (Optimizer): The optimizer to load the checkpoint into.
            learning_scheduler (lr_scheduler): The lr_scheduler to load the checkpoint into.

        Returns:
            None
        """
        checkpoint = torch.load(checkpoint_path)
        self.model = model.to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.loss = checkpoint['loss']
        if optimizer is not None:
            self.optimizer = optimizer
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if learning_scheduler is not None:
            self.lr_scheduler = learning_scheduler
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
        if start_epoch is not None:
            self.start_epoch = start_epoch
        return None
    
    def set_optimizer(self, optimizer):
        """
        Sets the optimizer to use for training.

        Args:
            optimizer (Optimizer): The optimizer to use for training.

        Returns:
            None
        """
        self.optimizer = optimizer
        return None
    
    def set_lr_scheduler(self, lr_scheduler):
        """
        Sets the lr_scheduler to use for training.

        Args:
            lr_scheduler (lr_scheduler): The lr_scheduler to use for training.

        Returns:
            None
        """
        self.lr_scheduler = lr_scheduler
        return None
    
    def train_model(
            self,
            epochs,
            data_loader,
            loss_function,
            model=None,
            optimizer=None,
            lr_scheduler=None,
            checkpoint_dir=None,
            checkpoint_interval=5,
            log_dir=None,
    ):
        """
        Trains self.model using optimizer and loss_function on the data in data_loader for epochs epochs.

        Args:
            epochs (int): The number of epochs to train for.
            data_loader (DataLoader): The DataLoader containing the training data.
            loss_function (Function): The loss function to use for training.
            model (nn.Module): The model to train.
            optimizer (Optimizer): The optimizer to use for training.
            lr_scheduler (lr_scheduler): The lr_scheduler to use for training.
            checkpoint_dir (str): The directory to save checkpoints to.
            checkpoint_interval (int): Create a checkpoint once every checkpoint_interval epochs.
            log_dir (str): The directory to save logs to.
        Returns:
            None
        """
        if model is not None:
            self.model = model.to(self.device)
        if optimizer is not None:
            self.optimizer = optimizer
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
        self.model.to(self.device)
        self.model.train()

        writer = SummaryWriter(log_dir=log_dir)
        scaler = torch.cuda.amp.GradScaler()

        num_batches = len(data_loader)
        for epoch in range(self.start_epoch, epochs+1):
            epoch_loss = 0
            min_loss = 100000
            with tqdm(data_loader, total=num_batches) as pbar:
                for b,batch in enumerate(pbar):
                    images = batch
                    times = torch.randint(0, self.num_timesteps,
                                           (self.batch_size,), device=self.device).long()
                    noisy_images, noise = self.forward_process(images, times[:,None,None,None])
                    predicted_noise = self.model(noisy_images, times)

                    self.optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        self.loss = loss_function(predicted_noise, noise)
                    scaler.scale(self.loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    batch_loss = self.loss.item()
                    epoch_loss += batch_loss
                    pbar.set_postfix({"Epoch:": epoch,
                                       "LR:":self.optimizer.param_groups[0]['lr'],
                                         "Loss:": epoch_loss/(b+1)})
            epoch_loss = epoch_loss/num_batches
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if checkpoint_dir is not None:
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler': self.lr_scheduler.state_dict(),
                    'loss': self.loss
                }
                if epoch%checkpoint_interval == 0:
                    torch.save(checkpoint_dict, os.path.join(checkpoint_dir,
                                                              "model_epoch_{}.pt".format(epoch)))
                    if epoch_loss < min_loss:
                        min_loss = epoch_loss
                        torch.save(checkpoint_dict, os.path.join(checkpoint_dir,
                                                                    "model_min_loss.pt"))
                    
            writer.add_scalar("Loss", epoch_loss, epoch)
            print("Epoch: {} Loss: {}".format(epoch, epoch_loss))
            print("LR: {}".format(self.optimizer.param_groups[0]['lr']))
        writer.close()

        return None
    
    def sample(
            self,
            num_images,
            variance_coeff=1
    ):
        """
        Runs the reverse diffusion process to generate a sample of num_images images from noise.

        Args:
            num_images (int): The number of images to generate.
            variance_coeff (float): Enables interpolation between the lower and upper bounds
                on reverse process entropy. variance_coeff = 0 is ideal for deterministic initial image,
                variance_coeff = 1 is ideal for normally distributed initial image.
                See https://arxiv.org/abs/2006.11239 for details.

        Returns:
            denoised_images (torch.Tensor): The denoised images after the reverse process.
        """
        self.model.to(self.device)
        self.model.eval()
        denoised_images = []
        with torch.no_grad():
            image = torch.randn(num_images, self.num_channels, self.image_size, self.image_size).to(self.device)
            for t in tqdm(reversed(range(1,self.num_timesteps))):
                t_batch = (torch.ones(num_images)*t).long().to(self.device)
                predicted_noise = self.model(image, t_batch)
                if t>1:
                    gaussian_noise = torch.randn_like(image)
                else:
                    gaussian_noise = 0
                alpha_batch = (torch.ones(num_images).to(self.device)*self.alpha[t])[:,None,None,None]
                alpha_bar_batch = (torch.ones(num_images).to(self.device)*self.alpha_bar[t])[:,None,None,None]
                alpha_bar_batch_prev = (torch.ones(num_images).to(self.device)*self.alpha_bar[t-1])[:,None,None,None]
                beta_batch = (torch.ones(num_images).to(self.device)*self.beta[t])[:,None,None,None]
                
                noise_coeff_1 = torch.sqrt((1- alpha_bar_batch_prev)/(1-alpha_bar_batch))
                noise_coeff_2 = 1
                noise_coeff = ((1-variance_coeff)*noise_coeff_1 + variance_coeff*noise_coeff_2)*torch.sqrt(beta_batch)
                image = 1/torch.sqrt(alpha_batch)*(image - ((1-alpha_batch)/torch.sqrt(1-alpha_bar_batch))*predicted_noise)\
                             + noise_coeff*gaussian_noise
                denoised_images.append(image)
        denoised_images = torch.stack(denoised_images, dim=0)
        denoised_images = denoised_images.permute(1,0,2,3,4)
        return denoised_images