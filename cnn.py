from torch import nn
import torch

class CNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        
        self.conv_layer1 = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv_layer2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.max_pool2 = nn.MaxPool3d(kernel_size = 2, stride = 2)


        self.conv_layer3 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3)
        self.conv_layer4 = nn.Conv3d(in_channels=16, out_channels=1, kernel_size=2)

        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(1, classes)
    
    def setAvgPoolDim(self, d, h, w):        
        return nn.AvgPool3d((d, h, w))

    def setLinearDim(self, d, h, w):
        return nn.Linear(d * h * w, 1, device='cuda')
    
    def forward(self, input):
        try:
            print("input shape: ", input.shape)
        except:
            print("Input is a list. Create a tensor input.")
            input = torch.cat(input)
            
        input = input.cuda()

        # ensure data types of images are torch.float32
        if input.dtype == torch.float64:
            input = input.to(torch.float32)
        
        out = self.conv_layer1(input)
        out = self.max_pool1(out)
        
        out = self.conv_layer2(out)
        out = self.relu1(out)
        out = self.max_pool2(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)        
        out = self.setAvgPoolDim(out.shape[2], out.shape[3], out.shape[4])(out)
        out = self.setLinearDim(out.shape[2], out.shape[3], out.shape[4])(out)
        out = self.relu2(out)
        
        out = self.fc(out)
        out = out.view(out.size(0), -1) 
        print("Final output shape:", out.shape)

        return out

