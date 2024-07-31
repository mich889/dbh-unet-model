import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), #doesnt need bias bc cancelled by batchnrom
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            #second conv
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x) 
    
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]): #feature output channels
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #downward path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        #upward path
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)) #adding skip connection in input
            self.ups.append(DoubleConv(feature*2, feature))
            
        self.bottleneck = DoubleConv(features[-1], features[-1]*2) 
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) #final conv that doesnt change number of channels

    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) #upsampling
            skip_connection = skip_connections[idx//2] #get skip connection

            if x.shape!=skip_connection.shape: #make sure divisible
                x = TF.resize(x, size=skip_connection.shape[2:]) #take out height and width

            concat_skip = torch.cat((skip_connection, x), dim=1) #concatenate along channel dimension
            x = self.ups[idx+1](concat_skip) #run thru double conv
            
        return self.final_conv(x)
    
def test():
    x = torch.randn((3, 1, 161, 161)) #divisble by 16
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
