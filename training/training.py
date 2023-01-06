import tqdm
import torch
import numpy as np


def training(num_epochs,net,optimizer,scheduler,dataloader_t,dataloader_v,device,path,warmup:int=0,future_pred:int=1,checkpoint=None):
    """
    Training for the network
    If a checkpoint dictionnary is used the state of network and optimizer will be initialized from the checkpoint
    Week W is the first week of future prediction.

    If the goal is to predict from week W to week W+future_pred-1 turnover using W-16 to W-1 data:
        The output y is a timeserie representing predicted turnover from week W-15 to W+future_pred-1, that is to say that the prediction 
        is y[:,-future_pred:].
        The output is a normalized output (as the input).



    At each epoch :
        - A training on the training set is performed using L1loss on the output[:,-(future_pred+warmup)], then the goal
            of the model is to predict turnovers from week W-warmup to week W+future_pred-1 using W-16 to W-1 data and 
            average annual turnover from W-16 to W+7
        - An evaluation on the training set is performed (compute the L1 loss on prediction from week W to week W+future_pred-1)
        - An evaluation on the validation set is performed (compute the L1 loss on prediction from week W to week W+future_pred-1)
        - A dictionnary D(epoch_N) is generated and stored. It contains:
            ->epoch value
            ->model state dict
            ->optimizer state dict
            ->List of all (from epoch 0 to epoch_N) the average loss calculated during for each epoch training on training set
            ->List of all (from epoch 0 to epoch_N) the average loss calculated during for each epoch evaluation on training set
            ->List of all (from epoch 0 to epoch_N) the average loss calculated during for each epoch evaluation on validation set

    Args:
        num_epochs(int): total number of epochs
        net(torch.nn.model): net used
        optimizer: optimizer used for training
        scheduler: scheduler used for training
        dataloader_t: dataloader of the training set
        dataloader_v:  dataloader of the validation set
        device: device used (CPU/GPU)
        path: path to a directory where the dictionnary D(epoch) will be stored
        warmup: number of week predictions before weel W used for training loss calculation.
        future_pred: number of future predictions to make (from 1 to 8).
        checkpoint: a dictionnary of type D(epoch) to use to start the training from a checkpoint  
    """
    if checkpoint!=None: #Load checpoint state
        starting_epoch=checkpoint["epoch"]+1
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        Loss_train_train=checkpoint["loss_train_train"]
        Loss_train_eval=checkpoint["loss_train_eval"]
        Loss_val_eval=checkpoint["loss_val_eval"]
    else: # Initialize
        starting_epoch=0
        Loss_train_train=[]
        Loss_train_eval=[]
        Loss_val_eval=[]
    for epoch in tqdm.tqdm(range(starting_epoch,num_epochs)):
        L_t_t=[]
        L_t_e=[]
        L_v_e=[]

        net.train()
        for i, data in (enumerate(dataloader_t, 0)): #Training
            net.zero_grad()
            y=net( data["x"].float().to(device),
                    data["annual_x"].float().to(device),
                    data["annual_y"].float().to(device),
                    future_pred)
            y=y[:,-(future_pred+warmup):]
            if warmup!=0:
                gt=torch.cat((data["x"][:,-warmup:].float().to(device),data["y"][:,0:future_pred].float().to(device)),dim=1)
            else:
                gt=data["y"][:,0:future_pred].float().to(device)
            loss=torch.nn.L1Loss()(y,gt)
            loss.backward()
            optimizer.step()
            L_t_t.append(loss.item()) #Loss of this batch training
            
        net.eval()

        for i, data in enumerate(dataloader_t, 0): #Evaluation on training set
            y=net( data["x"].float().to(device),
                    data["annual_x"].float().to(device),
                    data["annual_y"].float().to(device),
                    future_pred)
            y=y[:,-(future_pred):]
            gt=data["y"][:,0:future_pred].float().to(device)
            loss=torch.nn.L1Loss()(y,gt)
            L_t_e.append(loss.item()) #Loss of this batch evaluation

        for i, data in enumerate(dataloader_v, 0): #Evaluation on training set
            y=net( data["x"].float().to(device),
                    data["annual_x"].float().to(device),
                    data["annual_y"].float().to(device),
                    future_pred)
            y=y[:,-(future_pred):]
            gt=data["y"][:,0:future_pred].float().to(device)
            loss=torch.nn.L1Loss()(y,gt)
            L_v_e.append(loss.item()) #Loss of this batch evaluation

        scheduler.step()
        err_t_t=np.mean(L_t_t)
        err_t_e=np.mean(L_t_e)
        err_v_e=np.mean(L_v_e)
        Loss_train_train.append(err_t_t)
        Loss_train_eval.append(err_t_e)
        Loss_val_eval.append(err_v_e)
        print("Epoch : {} | \t Training : Loss {} |||  | \t Validation : Loss {} ||| ".format(epoch,err_t_e,err_v_e))
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train_train': Loss_train_train,
            'loss_train_eval': Loss_train_eval,
            'loss_val_eval':Loss_val_eval
            }, path+"checkpoint_{}.pth".format(epoch))