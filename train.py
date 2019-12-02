import torch
import tqdm
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, TensorDataset, DataLoader

def make_train_step(model, loss_fn, optimizer):
    '''
    Function to make a training step function.
    
    Parameters:
        model(Object): The model which is to be trained.
        loss_fn(Object) : The loss function which is to be used.
        optimizer(Object) : The optimiser to be used.
    Returns:
        Function : This can be called to train the model.
    '''
    def train_step(adj, h, y):
        model.train()
        yhat = model(adj, h)
        if type(loss_fn) != torch.nn.modules.loss.MSELoss:
            loss = loss_fn(yhat, y.long().squeeze())
        else:
            loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

def make_loader(adj_h, y, batch_size, valratio):
    '''
    Makes a train and validation data loader
    
    Parameters:
    adj_h (list) : list with two elements. The first element is the adjoint and the second element is the h
    y (list) : The label of the data
    
    Returns:
    (Dataloader object), (Dataloader object) : Dataloader for train and Dataloader for validation
    '''
    adj_tensor = torch.from_numpy(adj_h[0]).float()
    h_tensor = torch.from_numpy(adj_h[1]).float()
    y_tensor = torch.from_numpy(y).float()

    dataset = TensorDataset(adj_tensor, h_tensor, y_tensor)
    vallength = int(valratio*adj_h[0].shape[0])
    trainlength = adj_h[0].shape[0] - vallength
    
    train_dataset, val_dataset = random_split(dataset, [trainlength, vallength])
    if batch_size == None:
        train_loader = DataLoader(dataset=train_dataset, batch_size = trainlength, shuffle=True)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size = vallength)
    return train_loader, val_loader

def train_model(model, device, train_step, loss_fn, n_epochs, train_loader, val_loader):
    '''
    Function to run the validation and training.
    
    Parameters :
    model(nn.Module object) : The model which has to be trained
    device(torch.device object) : The device where it has to be trained
    train_step (function) : The training step function
    n_epoch (int) : The number of epochs to train the model
    train_loader (Dataloader object) : The Dataloader object for train
    val_loader (Dataloader object) : The Dataloader object for validation
    
    Returns :
    (list), (list), (list), (list) : training_losses, validation_losses, training_accuracy, validation_accuracy
    
    '''
    training_losses = []
    training_accuracy = []
    validation_losses = []
    validation_accuracy = []

    for epoch in range(n_epochs):
        batch_losses = []
        batch_acc = []
        for adj_batch, h_batch, y_batch in train_loader:
            #Put the batches on the gpu
            adj_batch = adj_batch.to(device)
            h_batch = h_batch.to(device)
            y_batch = y_batch.to(device)
            
            #Run the training by calling the train step and append the loss to the list. model.train is run in the model itself
            loss = train_step(adj_batch, h_batch, y_batch)
            batch_losses.append(loss)
            
            #Calculate the train accuracy with torh.nograd() and append in to the list
            if type(loss_fn) != torch.nn.modules.loss.MSELoss:
                with torch.no_grad():
                    model.eval()
                    yhat = model(adj_batch, h_batch)
                    train_acc = (y_batch.view(-1).float() == torch.argmax(yhat, axis = 1).float()).float().sum()/(y_batch.shape[0])
                    batch_acc.append(train_acc.item())
            else:
                batch_acc = batch_losses
                 
        training_losses.append(np.mean(batch_losses))
        training_accuracy.append(np.mean(batch_acc))

        #After one epoch check the validation score
        with torch.no_grad():
            val_losses = []
            val_acc = []
            for adj_val, h_val, y_val in val_loader:
                
                #Put the batches on the gpu
                adj_val = adj_val.to(device)
                h_val = h_val.to(device)
                y_val = y_val.to(device)
                
                #Calculate the model evaluation accuracy
                model.eval()
                yhat = model(adj_val, h_val) 
                if type(loss_fn) != torch.nn.modules.loss.MSELoss:
                    val_loss = loss_fn(yhat, y_val.long().squeeze())       
                    v_acc = (y_val.view(-1).float() == torch.argmax(yhat, axis = 1).float()).float().sum()/(y_val.shape[0])
                else:
                    val_loss = v_acc = loss_fn(yhat, y_val)  
                val_losses.append(val_loss.item())
                val_acc.append(v_acc.item())
            validation_losses.append(np.mean(val_losses))
            validation_accuracy.append(np.mean(val_acc))

        print(f"[{epoch+1}] Training loss: {training_losses[-1]:.6f}\t Validation loss: {validation_losses[-1]:.6f}")
        print(f"[{epoch+1}] Training Accuracy: {training_accuracy[-1]:.3f}\t Validation Accuracy: {validation_accuracy[-1]:.3f}")
    return training_losses, validation_losses, training_accuracy, validation_accuracy

