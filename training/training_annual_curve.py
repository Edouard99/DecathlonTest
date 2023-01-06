import tqdm
import torch
import numpy as np


def training(num_epochs,net,optimizer,scheduler,dataloader_t,device,path,checkpoint=None):
    if checkpoint!=None: #Load checpoint state
        starting_epoch=checkpoint["epoch"]+1
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        Loss_train_train=checkpoint["loss_train_train"]
        Loss_train_eval=checkpoint["loss_train_eval"]
    else: # Initialize
        starting_epoch=0
        Loss_train_train=[]
        Loss_train_eval=[]
    for epoch in tqdm.tqdm(range(starting_epoch,num_epochs)):
        L_t_t=[]
        L_t_e=[]

        net.train()
        for i, data in (enumerate(dataloader_t, 0)): #Training
            net.zero_grad()
            y=net(data["zod"].float().to(device),data["idr"].float().to(device),data["geo_pos"].float().to(device))
            gt=data["y"].float().to(device)
            loss=torch.nn.CosineSimilarity()(y,gt)
            loss=torch.mean(loss)
            loss=1-loss
            loss.backward()
            optimizer.step()
            L_t_t.append(loss.item()) #Loss of this batch training
            
        net.eval()

        for i, data in enumerate(dataloader_t, 0): #Evaluation on training set
            y=net(data["zod"].float().to(device),data["idr"].float().to(device),data["geo_pos"].float().to(device))
            gt=data["y"].float().to(device)
            loss=torch.nn.CosineSimilarity()(y,gt)
            loss=torch.mean(loss)
            loss=1-loss
            L_t_e.append(loss.item()) #Loss of this batch evaluation


        scheduler.step()
        err_t_t=np.mean(L_t_t)
        err_t_e=np.mean(L_t_e)
        Loss_train_train.append(err_t_t)
        Loss_train_eval.append(err_t_e)
        print("Epoch : {} | \t Training : Loss {} ||| ".format(epoch,err_t_e))
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train_train': Loss_train_train,
            'loss_train_eval': Loss_train_eval
            }, path+"checkpoint_{}.pth".format(epoch))