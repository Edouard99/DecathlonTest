import tqdm
import torch
import numpy as np


def training(num_epochs,net,optimizer,scheduler,dataloader_t,dataloader_v,device,path,warmup:int=0,future_pred:int=1,checkpoint=None):
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
            y=y[:,-(future_pred+warmup):]
            if warmup!=0:
                gt=torch.cat((data["x"][:,-warmup:].float().to(device),data["y"][:,0:future_pred].float().to(device)),dim=1)
            else:
                gt=data["y"][:,0:future_pred].float().to(device)
            loss=torch.nn.L1Loss()(y,gt)
            L_t_e.append(loss.item()) #Loss of this batch evaluation

        for i, data in enumerate(dataloader_v, 0): #Evaluation on training set
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