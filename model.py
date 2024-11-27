from torch import nn
import torch


def load_model_weights(model, state_dict_path):
    print(f'Loading from state_dict: {state_dict_path}')
    model.load_state_dict(torch.load(state_dict_path, weights_only=True))
    
    
def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
        
        
def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


class KoopmanViT(nn.Module):
    def __init__(self, vit, latent_size=768, batch_size=64, upsample_size=56):
        super(KoopmanViT, self).__init__()
        
        self.upsample_size = upsample_size
        
        self.encoder = vit
        self.koopman = nn.Linear(self.upsample_size**2, self.upsample_size**2)
        self.tanh = nn.Tanh()
        
        self.batch_size = batch_size
        
        # Fully connected layer to reshape latent size to 3136 (56x56)
        layers = []
        
        layers.append(nn.Linear(latent_size, 1400))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.35))
        layers.append(nn.Linear(1400, 2000))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(2000, self.upsample_size**2, bias=False))
        
        self.upscale = nn.Sequential(*layers)
        
        layers = []
        
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        layers.append(nn.Conv2d(1, 64, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.25))
        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.25))
        layers.append(nn.Conv2d(64, 32, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 3, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())
        
        self.decoder2 = nn.Sequential(*layers)
    
    def forward(self, x, y):
        x_encoded = self.upscale(self.encoder(x).pooler_output)
        
        x_recon = self.decoder2(x_encoded.view(64, 1, self.upsample_size, self.upsample_size))
                
        x_encoded_koopman = self.koopman(x_encoded)
        
        x_predicted = self.decoder2(x_encoded_koopman.view(64, 1, self.upsample_size, self.upsample_size))
        
        y_encoded = self.upscale(self.encoder(y).pooler_output)
        
        x_predicted_encoded = self.upscale(self.encoder(x_predicted).pooler_output)
        
        return x, y, x_predicted, y_encoded, x_predicted_encoded, x_encoded_koopman, x_recon
    
    
class KoopmanViTLoss(nn.Module):
    def __init__(self, a1, a2, a3, a4):
        super(KoopmanViTLoss, self).__init__()
        
        self.mse = lambda x, y: torch.mean(torch.mean(torch.linalg.norm(x - y, dim=(2, 3)), dim=1))
        self.mae = lambda x, y: torch.mean(torch.linalg.norm(x - y, dim=1, ord=1))
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        
    def forward(self, x, y, x_predicted, y_encoded, x_predicted_encoded, x_encoded_koopman, x_recon):
        
        # Reconstruction
        recon = self.mse(x, x_recon)
        
        # Prediction
        pred = self.mse(x_predicted, y)
        
        # Linearity
        lin = self.mae(x_encoded_koopman, y_encoded) 
        
        # additional noise reduction
        noise = self.mae(x_predicted_encoded, y_encoded)
                        
        return self.a1 * recon + self.a2 * pred + self.a3 * lin + self.a4 * noise
    
    
class ViTOperator(nn.Module):
    def __init__(self, vit, latent_size=768, batch_size=64, p=True):
        super(ViTOperator, self).__init__()
        
        self.encoder = vit
        self.up_state = 56
        self.batch_size = batch_size
        
        # Fully connected layer to reshape latent size to 3136 (56x56)
        layers = []
        
        layers.append(nn.Linear(latent_size, 1400))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.35 if p else 0.0))
        layers.append(nn.Linear(1400, 2000))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(2000, self.up_state**2))
        layers.append(nn.Tanh())
        
        self.upscale = nn.Sequential(*layers)
        
        layers = []
        
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        layers.append(nn.Conv2d(1, 32, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.25 if p else 0.0))
        layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        layers.append(nn.Conv2d(32, 16, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.25 if p else 0.0))
        layers.append(nn.Conv2d(16, 16, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 3, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())
        
        self.decoder2 = nn.Sequential(*layers)
    
    def forward(self, x, y):
        x_encoded = self.encoder(x).pooler_output
                
        x_predicted = self.decoder2(self.upscale(x_encoded).view(self.batch_size, 1, self.up_state, self.up_state))
                        
        return x_predicted, y
    
    
class ViTOperatorLarger(nn.Module):
    def __init__(self, vit, latent_size=768, batch_size=64, p=True):
        super(ViTOperatorLarger, self).__init__()
        
        self.encoder = vit
        self.up_state = 56
        self.batch_size = batch_size
        
        layers = []
        
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        layers.append(nn.Conv2d(1, 32, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.25 if p else 0.0))
        layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        layers.append(nn.Conv2d(32, 16, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.25 if p else 0.0))
        layers.append(nn.Conv2d(16, 16, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 3, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x, y):
        x_encoded = self.encoder(x).pooler_output
                
        x_predicted = self.decoder(x_encoded.view(self.batch_size, 1, self.up_state, self.up_state))
                        
        return x_predicted, y
    
    
class ViTOperatorFlex(nn.Module):
    def __init__(self, vit, latent_size=768, batch_size=64, p=True):
        super(ViTOperatorFlex, self).__init__()
        
        self.encoder = vit
        self.up_state = 56
        self.batch_size = batch_size
        
        # Fully connected layer to reshape latent size to 3136 (56x56)
        layers = []
        
        layers.append(nn.Linear(latent_size, self.up_state**2))
        layers.append(nn.ReLU())
        
        self.upscale = nn.Sequential(*layers)
        
        layers = []
        
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        layers.append(nn.Conv2d(1, 32, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.25 if p else 0.0))
        layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        layers.append(nn.Conv2d(32, 16, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.25 if p else 0.0))
        layers.append(nn.Conv2d(16, 16, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 3, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())
        
        self.decoder2 = nn.Sequential(*layers)
    
    def forward(self, x, y):
        x_encoded = self.encoder(x).pooler_output
                
        x_predicted = self.decoder2(self.upscale(x_encoded).view(self.batch_size, 1, self.up_state, self.up_state))
                        
        return x_predicted, y
    
    
class OperatorLoss(nn.Module):
    def __init__(self, a1):
        super(OperatorLoss, self).__init__()
        
        self.mse = lambda x, y: torch.mean(torch.mean(torch.linalg.norm(x - y, dim=(2, 3)), dim=1))
        self.mae = lambda x, y: torch.mean(torch.linalg.norm(x - y, dim=1, ord=1))
        self.a1 = a1
        
    def forward(self, x_predicted, y):
        # Prediction
        pred = self.mse(x_predicted, y)
                                
        return self.a1 * pred 