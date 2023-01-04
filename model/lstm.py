import torch
import torch.nn as nn

class LSTM_Turnover(nn.Module):
    def __init__(self, init_weights:bool, device, layer_number:int = 8):
        super(LSTM_Turnover, self).__init__()
        self.device=device
        self.layer_number=layer_number
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

        self.lstm=nn.LSTM(8,8,self.layer_number,batch_first=False,dropout=0.25)
        self.linears_end=nn.ModuleList([nn.Linear(8,1) for _ in range(0,8)])

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


    def forward(self,x: torch.Tensor,
                dep : torch.Tensor,idr: torch.Tensor,
                zod : torch.Tensor,hidden : torch.Tensor,sequence_len: int):
        dep_enc=self.dep_encoder(dep)
        zod_enc=self.zod_encoder(zod)
        idr_enc=self.idr_encoder(idr)
        hidden=torch.cat((hidden,dep_enc,zod_enc,idr_enc),dim=1)
        hidden=self.hidden_encoder(hidden)
        hidden=hidden.unsqueeze(0)
        c0=torch.zeros_like(hidden)
        x=x.unsqueeze(0)
        x=torch.cat([x for _ in range(0,sequence_len)],dim=0)
        hidden=torch.cat([hidden for _ in range(0,self.layer_number)],dim=0)
        c0=torch.cat([c0 for _ in range(0,self.layer_number)],dim=0)
        output_lstm, (hn, cn) = self.lstm(x, (hidden, c0))
        output_lstm=output_lstm[-8:].split(1)
        output=torch.randn((8,dep.shape[0],1)).to(self.device)
        for i,seq in enumerate(output_lstm):
            output[i]=nn.ReLU()(self.linears_end[i](seq))
        output=output.squeeze(-1)
        output=torch.permute(output,(1,0))
        return output


class LSTM_Turnoverv2(nn.Module):
    def __init__(self, init_weights:bool, hidden_size:int=48, num_of_layer:int=2):
        super(LSTM_Turnoverv2, self).__init__()
        self.hidden_size=hidden_size
        self.idr_encoder=nn.Sequential(
            nn.Linear(29,16),
            nn.Tanh(),
            nn.Linear(16,4),
            nn.Tanh()
        )
        self.zod_encoder=nn.Sequential(
            nn.Linear(8,6),
            nn.Tanh(),
            nn.Linear(6,4),
            nn.Tanh()
        )
        self.dep_encoder=nn.Sequential(
            nn.Linear(4,8),
            nn.Tanh(),
            nn.Linear(8,4),
            nn.Tanh()
        )
        self.input_encoder=nn.Sequential(
            nn.Linear(15,16),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(16,4),
            nn.Tanh()
        )
        self.output_encoder=nn.Sequential(
            nn.Linear(5,16),
            nn.Tanh(),
            nn.Linear(16,1)
            )
        self.add_lstm=nn.ModuleList([nn.LSTMCell(self.hidden_size,self.hidden_size) for _ in range(num_of_layer)])
        self.add_lstm[0]=nn.LSTMCell(5,self.hidden_size)
        self.linear=nn.Linear(self.hidden_size,1)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)
                elif isinstance(m,nn.LSTMCell):
                    for sub_m_name in m.state_dict().keys():
                        if 'weight' in sub_m_name:
                            nn.init.xavier_uniform_(m.state_dict()[sub_m_name])
                        if 'bias' in sub_m_name:
                            m.state_dict()[sub_m_name].data.fill_(0.01)


    def forward(self,x: torch.Tensor,
                dep : torch.Tensor,idr: torch.Tensor,
                zod : torch.Tensor,add_input : torch.Tensor,future_pred: int):
        device=x.get_device()
        dep_enc=self.dep_encoder(dep)
        zod_enc=self.zod_encoder(zod)
        idr_enc=self.idr_encoder(idr)
        add_input=torch.cat((add_input,dep_enc,zod_enc,idr_enc),dim=1)
        add_input=self.input_encoder(add_input)
        bactch_size=x.shape[0]
        outputs=[]
        h_t_list=[torch.zeros(bactch_size,self.hidden_size,dtype=torch.float32).to(device) for module in self.add_lstm]
        c_t_list=[torch.zeros(bactch_size,self.hidden_size,dtype=torch.float32).to(device) for module in self.add_lstm]
        for input_t in x.split(1,dim=1):
            input_t=torch.cat([input_t,add_input],dim=1)
            h_t_list[0],c_t_list[0]= self.add_lstm[0](input_t,(h_t_list[0],c_t_list[0]))
            for i,module in enumerate(self.add_lstm[1:]):
                h_t_list[i+1],c_t_list[i+1]= module(h_t_list[i],(h_t_list[i+1],c_t_list[i+1]))
            output=self.linear(h_t_list[-1])
            output=torch.cat([output,add_input],dim=1)
            output=self.output_encoder(output)
            outputs.append(output)
        for i in range(future_pred):
            output=torch.cat([output,add_input],dim=1)
            h_t_list[0],c_t_list[0]= self.add_lstm[0](output,(h_t_list[0],c_t_list[0]))
            for i,module in enumerate(self.add_lstm[1:]):
                h_t_list[i+1],c_t_list[i+1]= module(h_t_list[i],(h_t_list[i+1],c_t_list[i+1]))
            output=self.linear(h_t_list[-1])
            output=torch.cat([output,add_input],dim=1)
            output=self.output_encoder(output)
            outputs.append(output)
        outputs=torch.cat(outputs, dim=1)
        return outputs

class Modelev12(nn.Module):
    def __init__(self, init_weights:bool, hidden_size:int=48, num_of_layer:int=2):
        super(Modelev12, self).__init__()
        self.hidden_size=hidden_size
        
        self.add_lstm=nn.ModuleList([nn.LSTMCell(self.hidden_size,self.hidden_size) for _ in range(num_of_layer)])
        self.add_lstm[0]=nn.LSTMCell(1,self.hidden_size)
        self.linear=nn.Linear(self.hidden_size,1)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)
                elif isinstance(m,nn.LSTMCell):
                    for sub_m_name in m.state_dict().keys():
                        if 'weight' in sub_m_name:
                            nn.init.xavier_uniform_(m.state_dict()[sub_m_name])
                        if 'bias' in sub_m_name:
                            m.state_dict()[sub_m_name].data.fill_(0.01)


    def forward(self,x: torch.Tensor,future_pred: int):
        device=x.get_device()
        bactch_size=x.shape[0]
        outputs=[]
        h_t_list=[torch.zeros(bactch_size,self.hidden_size,dtype=torch.float32).to(device) for module in self.add_lstm]
        c_t_list=[torch.zeros(bactch_size,self.hidden_size,dtype=torch.float32).to(device) for module in self.add_lstm]
        for input_t in x.split(1,dim=1):
            h_t_list[0],c_t_list[0]= self.add_lstm[0](input_t,(h_t_list[0],c_t_list[0]))
            for i,module in enumerate(self.add_lstm[1:]):
                h_t_list[i+1],c_t_list[i+1]= module(h_t_list[i],(h_t_list[i+1],c_t_list[i+1]))
            output=self.linear(h_t_list[-1])
            outputs.append(output)
        for i in range(future_pred):
            h_t_list[0],c_t_list[0]= self.add_lstm[0](output,(h_t_list[0],c_t_list[0]))
            for i,module in enumerate(self.add_lstm[1:]):
                h_t_list[i+1],c_t_list[i+1]= module(h_t_list[i],(h_t_list[i+1],c_t_list[i+1]))
            output=self.linear(h_t_list[-1])
            outputs.append(output)
        outputs=torch.cat(outputs, dim=1)
        return outputs

class Modelev34(nn.Module):
    def __init__(self, init_weights:bool, hidden_size:int=48, num_of_layer:int=2):
        super(Modelev34, self).__init__()
        self.hidden_size=hidden_size
        
        self.add_lstm=nn.ModuleList([nn.LSTMCell(self.hidden_size,self.hidden_size) for _ in range(num_of_layer)])
        self.add_lstm[0]=nn.LSTMCell(5,self.hidden_size)
        self.linear=nn.Linear(self.hidden_size,1)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)
                elif isinstance(m,nn.LSTMCell):
                    for sub_m_name in m.state_dict().keys():
                        if 'weight' in sub_m_name:
                            nn.init.xavier_uniform_(m.state_dict()[sub_m_name])
                        if 'bias' in sub_m_name:
                            m.state_dict()[sub_m_name].data.fill_(0.01)


    def forward(self,x: torch.Tensor,future_pred: int):
        device=x.get_device()
        bactch_size=x.shape[0]
        outputs=[]
        h_t_list=[torch.zeros(bactch_size,self.hidden_size,dtype=torch.float32).to(device) for module in self.add_lstm]
        c_t_list=[torch.zeros(bactch_size,self.hidden_size,dtype=torch.float32).to(device) for module in self.add_lstm]
        for input_t in x.split(1,dim=1):
            h_t_list[0],c_t_list[0]= self.add_lstm[0](input_t,(h_t_list[0],c_t_list[0]))
            for i,module in enumerate(self.add_lstm[1:]):
                h_t_list[i+1],c_t_list[i+1]= module(h_t_list[i],(h_t_list[i+1],c_t_list[i+1]))
            output=self.linear(h_t_list[-1])
            outputs.append(output)
        for i in range(future_pred):
            h_t_list[0],c_t_list[0]= self.add_lstm[0](output,(h_t_list[0],c_t_list[0]))
            for i,module in enumerate(self.add_lstm[1:]):
                h_t_list[i+1],c_t_list[i+1]= module(h_t_list[i],(h_t_list[i+1],c_t_list[i+1]))
            output=self.linear(h_t_list[-1])
            outputs.append(output)
        outputs=torch.cat(outputs, dim=1)
        return outputs

class Modelev5(nn.Module):
    def __init__(self, init_weights:bool, hidden_size:int=48, num_of_layer:int=2):
        super(Modelev34, self).__init__()
        self.hidden_size=hidden_size
        
        self.add_lstm=nn.ModuleList([nn.LSTMCell(self.hidden_size,self.hidden_size) for _ in range(num_of_layer)])
        self.add_lstm[0]=nn.LSTMCell(16,self.hidden_size)
        self.linear=nn.Linear(self.hidden_size,1)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)
                elif isinstance(m,nn.LSTMCell):
                    for sub_m_name in m.state_dict().keys():
                        if 'weight' in sub_m_name:
                            nn.init.xavier_uniform_(m.state_dict()[sub_m_name])
                        if 'bias' in sub_m_name:
                            m.state_dict()[sub_m_name].data.fill_(0.01)


    def forward(self,x: torch.Tensor,future_pred: int):
        device=x.get_device()
        bactch_size=x.shape[0]
        outputs=[]
        h_t_list=[torch.zeros(bactch_size,self.hidden_size,dtype=torch.float32).to(device) for module in self.add_lstm]
        c_t_list=[torch.zeros(bactch_size,self.hidden_size,dtype=torch.float32).to(device) for module in self.add_lstm]

        h_t_list[0],c_t_list[0]= self.add_lstm[0](x,(h_t_list[0],c_t_list[0]))
        for i,module in enumerate(self.add_lstm[1:]):
            h_t_list[i+1],c_t_list[i+1]= module(h_t_list[i],(h_t_list[i+1],c_t_list[i+1]))
        output=self.linear(h_t_list[-1])
        outputs.append(output)

        input_t=x
        for i in range(future_pred):
            h_t_list[0],c_t_list[0]= self.add_lstm[0](input_t,(h_t_list[0],c_t_list[0]))
            for i,module in enumerate(self.add_lstm[1:]):
                h_t_list[i+1],c_t_list[i+1]= module(h_t_list[i],(h_t_list[i+1],c_t_list[i+1]))
            output=self.linear(h_t_list[-1])
            outputs.append(output)
            input_t=torch.cat([input_t[1:],output],dim=1)
        outputs=torch.cat(outputs, dim=1)
        return outputs
