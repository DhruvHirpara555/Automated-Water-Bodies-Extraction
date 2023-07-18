

import torch
import torch.nn as nn
import torch.nn.functional as F

class MDIT(nn.Module):
    def __init__(self, min_width=4):
        super(MDIT, self).__init__()

        self.fl = nn.Sequential(
            nn.Conv2d(4, min_width, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(min_width)
        )

        self.encoding_units = nn.ModuleList()
        for i in range(4):
            self.encoding_units.append(nn.Sequential(
                nn.Conv2d(min_width*(4**i), min_width * (4**(i+1)), kernel_size=5, stride=2, padding=2 ,bias=False),
                nn.BatchNorm2d(min_width * (4**(i+1))),
                nn.ReLU()
            ))


        self.encod_residual = nn.ModuleList()
        for i in range(4):
            self.encod_residual.append(nn.Sequential(
                nn.Conv2d(min_width * (4**(i+1)), min_width * (4**(i+1)), kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(min_width * (4**(i+1))),
                nn.ReLU()
            ))

        # self.decoding_units = nn.ModuleList()
        self.decod_residual1 = nn.ModuleList()
        for i in range(3,-1,-1):
            self.decod_residual1.append(nn.Sequential(
                nn.Conv2d(min_width * (4**(i)), min_width * (4**(i)), kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(min_width * (4**(i))),
                nn.ReLU()
            ))

        self.decod_residual2 = nn.ModuleList()
        for i in range(3,-1,-1):
            self.decod_residual2.append(nn.Sequential(
                nn.Conv2d(min_width * (4**(i)), min_width * (4**(i)), kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(min_width * (4**(i))),
                nn.ReLU()
            ))







        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )





        self.last = nn.Sequential(
            nn.Conv2d(min_width, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        skip_connections = []
        x = self.fl(x)
        skip_connections.append(x)


        for i in range(4):
            x_1 = self.encoding_units[i](x)
            x_2 = self.encod_residual[i](x_1)
            x = x_1 + x_2
            skip_connections.append(x)



        x_1 = self.bottleneck1(x)
        x_2 = self.bottleneck2(x_1)
        x = x_1 + x_2


        for i in range(4):
            x = skip_connections.pop() + x
            # print(x.shape)
            x = x.reshape(x.shape[0], x.shape[1]//4, 2*x.shape[2], 2*x.shape[3])
            # print(x.shape)
            x_1 = self.decod_residual1[i](x)
            x_2 = self.decod_residual2[i](x_1)
            x = x_1 + x_2



        x = skip_connections.pop() + x
        x = self.last(x)
        x = x.squeeze(1)
        return x