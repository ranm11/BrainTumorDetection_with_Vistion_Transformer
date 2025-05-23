import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import os
import numpy as np
import random
import math
import json
from functools import partial
from PIL import Image
## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')
#%matplotlib inline
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
#import seaborn as sns
#sns.reset_orig()

## Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# DATASET_PATH = "data"
#CHECKPOINT_PATH = "saved_models"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
#tutorial 15
#pl.seed_everything(42)

def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x

class AttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, 
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.attn_out = None
                
    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        self.attn_out , _ = self.attn(inp_x, inp_x, inp_x, need_weights=True, average_attn_weights=False)
        x = x + self.attn_out
        x = x + self.linear(self.layer_norm_2(x))
        return x

    def get_attention_maps(self,x):
        inp_x = self.layer_norm_1(x)
        attn_map = self.attn(inp_x, inp_x, inp_x, need_weights=True, average_attn_weights=False)[1]
        return attn_map
    
class VisionTransformer(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and 
                      on the input encoding
        """
        super().__init__()
        
        self.patch_size = patch_size
        
        # Layers/Networks
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))
    
    
    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)
        
        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]
        
        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        
        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out

    def get_positional_encoding(self,img):
        x = img_to_patch(img, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)
        pe = self.pos_embedding[:,:T]
        return pe

    def get_attention_maps(self,img, mask=None):
        x = img_to_patch(img, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)
 
        x = x + self.pos_embedding[:,:T]
 
        x = self.dropout(x)
        x = x.transpose(0, 1)
        attn_map=[]
        for i, layer in enumerate(self.transformer):
            if isinstance(layer, AttentionBlock):
                attn_map.append(layer.get_attention_maps(x))
        return torch.stack(attn_map,dim=0)

class ViT(pl.LightningModule):
    
    def __init__(self, model_kwargs,lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)
        # self.example_input_array = next(iter(train_loader))[0]
    def get_attention_maps(self,img):
        return self.model.get_attention_maps(img)
    
    def get_positional_encoding(self,img):
        encod_block = self.model.get_positional_encoding(img)
        pe = encod_block[0].detach().numpy()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,3))
        pos = ax.imshow(pe, cmap="RdGy", extent=(1,pe.shape[1]+1,pe.shape[0]+1,1))
        fig.colorbar(pos, ax=ax)
        ax.set_xlabel("Position in sequence")
        ax.set_ylabel("Hidden dimension")
        ax.set_title("Positional encoding for embedded input")
        ax.set_xticks([1]+[i*10 for i in range(1,1+pe.shape[1]//10)])
        ax.set_yticks([1]+[i*10 for i in range(1,1+pe.shape[0]//10)])
        filename = "positional_encoding.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        return [optimizer], [lr_scheduler]   
    
    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)  #this function internally apply the softMax activation
        # loss = F.binary_cross_entropy_with_logits(preds, labels)  # simply binary 
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        self.log(f'{mode}_loss', loss,prog_bar=True, logger=True)
        self.log(f'{mode}_acc', acc,prog_bar=True, logger=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
