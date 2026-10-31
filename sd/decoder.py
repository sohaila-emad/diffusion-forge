import torch 
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int ):
        super().__init__()
        self.channels = channels
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: (batch_size, channels, hieght, width)
        residual = x

        #we are changing the shape of the input tensor to (batch_size, hieght*width, channels)
        #to apply attention mechanism because its originally
        #applied to words(sequence data) so it expects(batch_size, sequence_length, embedding_dim)
        n, c, h, w = x.shape
        x = x.view(n, c, h * w).permute(0, 2, 1)  # (batch_size, hieght*width, channels)

        x = self.attention(x)  # Apply attention
        #no return it back to the original shape 
        x = x.permute(0, 2, 1).view(n, c, h, w)  # (batch_size, channels, hieght, width)

        x += residual
        return x

class VAE_ResidualBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        self.residual_layer = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        x += self.residual_layer(residual)

        return x

class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            #(batch_size, 4, hieght/8, width/8)--> (batch_size, 512, hieght/8, width/8)
            nn.Conv2d(4, 4, kernel_size = 1, padding = 0),
            #(batch_size, 4, hieght/8, width/8)--> (batch_size, 512, hieght/8, width/8)
            nn.Conv2d(4, 512, kernel_size = 3, padding = 1),

            #(batch_size, 512, hieght/8, width/8)--> (batch_size, 512, hieght/8, width/8)
            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            #(batch_size, 512, hieght/8, width/8)--> (batch_size, 512, hieght/4, width/4)
            nn.Upsample(scale_factor = 2),
            #(batch_size ,512 , hieght /2 , width /2 )--> (batch_size ,256 , hieght /2 , width /2 )
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),

            VAE_ResidualBlock(512 ,512 ),
            VAE_ResidualBlock(512 ,512 ),
            VAE_ResidualBlock(512 ,512 ),

            #(batch_size ,512 , hieght /4 , width /4 )--> (batch_size ,512 , hieght /2 , width /2 )
            nn.Upsample(scale_factor = 2),

            #(batch_size ,512 , hieght /2 , width /2 )--> (batch_size ,512 , hieght , width )
            nn.Conv2d(512 ,512 , kernel_size =3 , padding =1 ),

            VAE_ResidualBlock(512 ,256 ),
            VAE_ResidualBlock(256 ,256 ),
            VAE_ResidualBlock(256 ,256 ),

            nn.Upsample(scale_factor = 2),

            #(batch_size ,256 , hieght , width )--> (batch_size ,256 , hieght , width )
            nn.Conv2d(256 ,256 , kernel_size =3 , padding =1 ),

            VAE_ResidualBlock(256 ,128 ),
            VAE_ResidualBlock(128 ,128 ),
            VAE_ResidualBlock(128 ,128 ),

            nn.GroupNorm(32 ,128 ),
            nn.SiLU(),

            #just to get the output channels to 3 for RGB images
            nn.Conv2d(128 ,3 , kernel_size =3 , padding =1 ) 
      
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: (batch_size, 4, hieght/8, width/8)

        x/= 0.18215 #scaling factor to bring the output to the original range of the input images

        for module in self:
            x = module(x)
        #(batch_size, 3, hieght, width)
        return x


