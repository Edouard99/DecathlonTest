import torch
import torch.nn as nn

class Curvegen(nn.Module):
    def __init__(self, init_weights:bool, nf:int):
        super(Curvegen, self).__init__()
        self.nf=nf
        self.idr_encoder=nn.Sequential(
            nn.Linear(29,32),
            nn.ReLU(),
            nn.Linear(32,4),
            nn.ReLU()
        )
        self.zod_encoder=nn.Sequential(
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,4),
            nn.ReLU()
        )
        self.input_encoder=nn.Sequential(
            #nn.Linear(2,48),
            #nn.ReLU(),
            nn.Linear(2,16),
            nn.ReLU()
        )
        self.deconvnet=nn.Sequential(
            #Nx1x16
            nn.Conv1d(1,nf,1,1),
            #Nxnfx16
            DeconvModule(nf,nf,3,2,1,1),
            #Nxnf*2x32
            DeconvModule(nf*2,nf*2,3,2,1,1),
            #Nxnf*4x64
            DeconvModule(nf*4,nf*4,3,2,1,1),
            #Nxnf*8x128
            nn.Conv1d(nf*8,1,1,1),
            #Nx1x128
            nn.Sigmoid()
            #Nx1x128
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)
                elif isinstance(m,nn.ConvTranspose1d):
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                    nn.init.constant_(m.bias.data, 0)
                elif isinstance(m,nn.BatchNorm1d):
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0)


    def forward(self,zod: torch.Tensor,idr: torch.Tensor,geo_pos: torch.Tensor):
        #zod=self.zod_encoder(zod)
        #idr=self.idr_encoder(idr)
        #input_deconv=torch.cat([zod,idr,geo_pos],dim=1).to(zod.get_device())
        #input_deconv=self.input_encoder(input_deconv)
        input_deconv=self.input_encoder(geo_pos)
        input_deconv=input_deconv.unsqueeze(1)
        output=self.deconvnet(input_deconv)
        output=output.squeeze(1)
        return output

class DeconvModule(nn.Module):
    def __init__(self, in_c,out_c,k,s,p,op):
        super(DeconvModule, self).__init__()
        self.convt=nn.ConvTranspose1d(in_c,out_c,k,s,p,op)
        self.relu=nn.ReLU()
    def forward(self,x:torch.Tensor):
        y=self.convt(x)
        y=self.relu(y)
        x=nn.Upsample(y.shape[2])(x)
        y=torch.cat([x,y],dim=1)
        return y
