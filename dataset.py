import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch import nn


def main(x_train_data, y_train_data, x_valid_data, y_valid_data, device, batch_size):
    train_dataset = collidingBubbles(x_train_data, y_train_data, device, load_instances, preprocess)
    valid_dataset = collidingBubbles(x_valid_data, y_valid_data,  device, load_instances, preprocess)
    
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                      num_workers=32, drop_last=True, pin_memory=True)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, 
                      num_workers=32, drop_last=True, pin_memory=True)
    
    return train_dl, valid_dl


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std_range=(0.0, 0.1)):
        self.mean = mean
        self.std_range = std_range

    def __call__(self, tensor):
        if torch.rand(1) < 0.5:
            noise = torch.normal(self.mean, (self.std_range[1] - self.std_range[0]) * torch.rand(1).item(), size=tensor.size())
            return tensor + noise
        return tensor
    
class AddSaltAndPepperNoise(object):
    def __init__(self, prob=0.01):
        self.prob = prob

    def __call__(self, tensor):
        # Add salt-and-pepper noise
        if torch.rand(1) < 0.5:
            mask = torch.rand(tensor.shape)
            tensor[mask < self.prob / 2] = 0  # Salt
            tensor[mask > 1 - self.prob / 2] = 1  # Pepper
        return tensor

class RandomGaussianBlur(object):
    def __init__(self, kernel_size_range=(3, 7), sigma_range=(0.1, 2.0)):
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range
        
    def __call__(self, image):
        if torch.rand(1) > 0.5:
            return image
        # Randomly choose kernel size and sigma within the specified range
        kernel_size = random.choice([k for k in range(self.kernel_size_range[0], self.kernel_size_range[1] + 1, 2)])
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])

        # Generate a Gaussian kernel for blurring
        kernel = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (kernel / sigma).pow(2))
        kernel = kernel / kernel.sum()  # Normalize

        # Apply the 1D kernel to blur the image in both horizontal and vertical directions
        kernel_2d = kernel[:, None] * kernel[None, :]  # Create a 2D Gaussian kernel
        kernel_2d = kernel_2d.expand(image.shape[0], 1, kernel_size, kernel_size)

        # Apply the 2D convolution to each channel
        image = image.unsqueeze(0)  # Add batch dimension for conv2d
        blurred_image = F.conv2d(image, kernel_2d, padding=kernel_size // 2, groups=image.shape[1])

        return blurred_image.squeeze(0)  # Remove batch dimension

aug_pipeline = v2.Compose([
    AddGaussianNoise(),
    AddSaltAndPepperNoise(),
    RandomGaussianBlur(),
    v2.RandomHorizontalFlip(p=0.5)
])


def indepent_channel_select(x_vec, channels=[3], seq_len=8):  
    tensor = torch.zeros((len(channels)*seq_len,) + x_vec.shape[1:])
    
    for ind in range(x_vec.shape[0] // 4):
        base_idx = ind * 4
        for idx, ch in enumerate(channels):
            if ch == 1:
                tensor[idx+ind*len(channels)] = x_vec[base_idx+1] / x_vec[base_idx]
            if ch == 2:
                tensor[idx+ind*len(channels)] = x_vec[base_idx+2] / x_vec[base_idx]
            if ch == 3:
                tensor[idx+ind*len(channels)] = x_vec[base_idx+3] / x_vec[base_idx]
    
    return tensor


def load_instances(data_instance, upsample_size=224, dtype=torch.float32, channels=4, seq_len=8, channel_select=[3]):
    np_instances = np.zeros((data_instance.shape[0], channels*10000))
    
    for idx, instance in enumerate(data_instance):
        np_instances[idx, :] = np.load(instance)
    
    torch_instances = torch.from_numpy(np_instances).to(dtype=dtype)
    
    if upsample_size is not None:
        inpt = nn.functional.interpolate(torch_instances.view(1, seq_len*channels, 100, 100), size=(upsample_size, upsample_size), mode='bicubic')
        return indepent_channel_select(inpt.view(channels*seq_len, upsample_size, upsample_size), seq_len=seq_len, channels=channel_select)
    
    return indepent_channel_select(inpt.view(channels*seq_len, 100, 100), seq_len=seq_len, channels=channel_select)  


def preprocess(input_tensor, target_tensor):
    # Compute the min and max values for each image in the batch
    in_mins = torch.amin(input_tensor, dim=(1, 2), keepdim=True)
    in_maxes = torch.amax(input_tensor, dim=(1, 2), keepdim=True)
    
    tar_mins = torch.amin(target_tensor, dim=(1, 2), keepdim=True)
    tar_maxes = torch.amax(target_tensor, dim=(1, 2), keepdim=True)
    
    # Apply min-max scaling to [0, 1]
    standard_input = (input_tensor - in_mins) / (in_maxes - in_mins + 1e-8)  # Adding a small epsilon to avoid division by zero
    standard_target = (target_tensor - tar_mins) / (tar_maxes - tar_mins + 1e-8)
    
    return standard_input, standard_target


class collidingBubbles(Dataset):
    def __init__(self, x, y, device, loading_func, preprocess_func, data_aug=None, channel_select=None):
        self.device = device
        self.x, self.y = x, y
        self.loading_func = loading_func
        self.preprocess_func = preprocess_func
        self.channel_selection = channel_select
        self.data_aug = data_aug

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_instance, y_instance = self.x[idx], self.y[idx]
                
        inpt, target = self.loading_func(x_instance, seq_len=1, channel_select=[1, 2, 3]), self.loading_func(y_instance, seq_len=1, channel_select=[1, 2, 3])
        inpt, target = self.preprocess_func(inpt, target)
        
        if self.data_aug is not None:
            t = self.data_aug(torch.cat([inpt, target], axis=0))
            inpt_aug, target_aug = t[:3, :, :], t[3:, :, :]

            return inpt, target, inpt_aug, target_aug
        
        return inpt, target