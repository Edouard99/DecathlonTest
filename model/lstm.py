import torch
import torch.nn as nn

class FaceNet(nn.Module):
    def __init__(self,init_weights:bool):


        super(FaceNet, self).__init__()
        self.conv1=ConvModule(3,3,1,1,0)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self,x:torch.Tensor):
        #Nx3x56x56
        x=self.conv1(x)
        return x

class LSTM_Turnover(nn.Module):
    def __init__(self, init_weights:bool):
        super(LSTM_Turnover, self).__init__()
        self.idr_encoder=nn.Sequential(
            nn.Linear(29,64),
            nn.ReLU(),
            nn.Linear(64,4),
            nn.ReLU()
        )
        self.zod_encoder=nn.Sequential(
            nn.Linear(8,32),
            nn.ReLU(),
            nn.Linear(32,4),
            nn.ReLU()
        )
        self.dep_encoder=nn.Sequential(
            nn.Linear(4,32),
            nn.ReLU(),
            nn.Linear(32,4),
            nn.ReLU()
        )
        self.hidden_encoder=nn.Sequential(
            nn.Linear(15,64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,8),
            nn.ReLU()
        )

        self.lstm=nn.LSTM(8,8,8,batch_first=True)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)
                elif isinstance(m,nn.LSTM):
                    for sub_m_name in m.state_dict().keys():
                        if 'weight' in sub_m_name:
                            nn.init.xavier_uniform_(m.state_dict()[sub_m_name])
                        if 'bias' in sub_m_name:
                            m.state_dict()[sub_m_name].data.fill_(0.01)


    def forward(self,x: torch.Tensor,y: torch.Tensor,
                dep : torch.Tensor,idr: torch.Tensor,
                zod : torch.Tensor,hidden : torch.Tensor,sequence_len: int):
        dep_enc=self.dep_encoder(dep)
        zod_enc=self.zod_encoder(zod)
        idr_enc=self.idr_encoder(idr)
        print((dep_enc.shape , zod_enc.shape , idr_enc.shape))
        hidden=torch.cat((hidden,dep_enc,zod_enc,idr_enc),dim=1)
        print(hidden.shape)
        hidden=self.hidden_encoder(hidden)
        c0=torch.zeros_like(hidden)
        x=x.unsqueeze(1)
        print(x.shape)
        x=torch.cat([x for _ in range(0,sequence_len)],dim=1)
        print(hidden.shape)
        print(x.shape)
        print(c0.shape)
        output, (hn, cn) = self.lstm(x, (hidden, c0))
        print(output.shape)
        return output