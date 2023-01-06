import torch
import torch.nn as nn

class LSTM_Turnover(nn.Module):
    """
    LSTM model that predicts from 1 to 8 weeks next turnovers given a timeserie of last 16 weeks, a timeserie of last 
    16 weeks portion of average annual turnover, a timeserie of next 8 weeks portion of average annual turnover.
    The model can use multiple lstm layer and can be initialized with xavier init.
    If the goal is to predict turnovers from week W to week W+future_pred-1 using W-16 to W-1 data and average annual turnover from W-16 to W+7:
        The output y is a timeserie representing predicted turnover from week W-15 to W+future_pred-1, that is to say that the prediction 
        is y[:,-future_pred:].
        The output is a normalized output (as the input).
    Args:
        -init_weights(bool): if True the weights will be initialized with xavier init.
        -hidden_size(int): size of the hidden state in LSTM cells
        -num_of_layer(int): number of lstm layers
    """
    def __init__(self, init_weights:bool, hidden_size:int=64, num_of_layer:int=4):
        super(LSTM_Turnover, self).__init__()
        self.hidden_size=hidden_size
        
        self.lstm_cells=nn.ModuleList([nn.LSTMCell(self.hidden_size,self.hidden_size) for _ in range(num_of_layer)])
        self.lstm_cells[0]=nn.LSTMCell(2,self.hidden_size)
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


    def forward(self,x: torch.Tensor,annual_x: torch.Tensor, annual_y: torch.Tensor,future_pred: int):
        """
        Make a prediction

        Args:
            -x(torch.Tensor): normalized turnover timeserie of shape (Batchsize,16)
            -annual_x(torch.Tensor): normalized average annual turnover timeserie of shape (Batchsize,16)
            -annual_y(torch.Tensor): normalized average annual turnover timeserie of shape (Batchsize,8)
            -future_pred(int): make a prediction from week W to week W+future_pred-1
        """
        device=x.device
        bactch_size=x.shape[0]
        outputs=[]
        h_t_list=[torch.zeros(bactch_size,self.hidden_size,dtype=torch.float32).to(device) for module in self.lstm_cells]
        c_t_list=[torch.zeros(bactch_size,self.hidden_size,dtype=torch.float32).to(device) for module in self.lstm_cells]
        for input_x,input_ax in zip(x.split(1,dim=1)[:-1],annual_x.split(1,dim=1)[1:]):
            h_t_list[0],c_t_list[0]= self.lstm_cells[0](torch.cat((input_x,input_ax),dim=1),(h_t_list[0],c_t_list[0]))
            for i,module in enumerate(self.lstm_cells[1:]):
                h_t_list[i+1],c_t_list[i+1]= module(h_t_list[i],(h_t_list[i+1],c_t_list[i+1]))
            output=self.linear(h_t_list[-1])
            output=nn.Tanh()(output)
            output=input_ax+output
            outputs.append(output)

        output=x.split(1,dim=1)[-1]
        annual_y=annual_y.split(1,dim=1)
        if future_pred>len(annual_y):
            future_pred=len(annual_y)
        for pred in range(future_pred):
            h_t_list[0],c_t_list[0]= self.lstm_cells[0](torch.cat((output,annual_y[pred]),dim=1),(h_t_list[0],c_t_list[0]))
            for i,module in enumerate(self.lstm_cells[1:]):
                h_t_list[i+1],c_t_list[i+1]= module(h_t_list[i],(h_t_list[i+1],c_t_list[i+1]))
            output=self.linear(h_t_list[-1])
            output=nn.Tanh()(output)
            output=annual_y[pred]+output
            outputs.append(output)
        outputs=torch.cat(outputs, dim=1)
        return outputs


class LSTM_Turnoverv0(nn.Module):
    """
    Old model not using annual average turnover
    
    """
    def __init__(self, init_weights:bool, hidden_size:int=48, num_of_layer:int=2):
        super(LSTM_Turnoverv0, self).__init__()
        self.hidden_size=hidden_size
        
        self.lstm_cells=nn.ModuleList([nn.LSTMCell(self.hidden_size,self.hidden_size) for _ in range(num_of_layer)])
        self.lstm_cells[0]=nn.LSTMCell(2,self.hidden_size)
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
        h_t_list=[torch.zeros(bactch_size,self.hidden_size,dtype=torch.float32).to(device) for module in self.lstm_cells]
        c_t_list=[torch.zeros(bactch_size,self.hidden_size,dtype=torch.float32).to(device) for module in self.lstm_cells]
        for input_x in x.split(1,dim=1):
            h_t_list[0],c_t_list[0]= self.lstm_cells[0](input_x,(h_t_list[0],c_t_list[0]))
            for i,module in enumerate(self.lstm_cells[1:]):
                h_t_list[i+1],c_t_list[i+1]= module(h_t_list[i],(h_t_list[i+1],c_t_list[i+1]))
            output=self.linear(h_t_list[-1])
            outputs.append(output)
        for pred in range(future_pred):
            h_t_list[0],c_t_list[0]= self.lstm_cells[0](output,(h_t_list[0],c_t_list[0]))
            for i,module in enumerate(self.lstm_cells[1:]):
                h_t_list[i+1],c_t_list[i+1]= module(h_t_list[i],(h_t_list[i+1],c_t_list[i+1]))
            output=self.linear(h_t_list[-1])
            outputs.append(output)
        outputs=torch.cat(outputs, dim=1)
        return outputs

