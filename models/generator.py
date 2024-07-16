import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_noise_vector_size,feature_map_depth,number_of_channels) -> None:
        super(Generator,self).__init__()
        self.main ==  nn.Sequential(
                nn.ConvTranspose2d(in_channels=input_noise_vector_size,out_channels=feature_map_depth*8,kernel_size=4,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(number_of_channels*8),
                nn.nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=feature_map_depth*8,out_channels=feature_map_depth*4,kernel_size=4,stride=1,padding=0,bias=False)
                ,nn.BatchNorm2d(input_noise_vector_size*4)
                ,
                nn.ReLU(True),
