import math
import sys
import torch


def progress(loss, epoch, batch, batch_size, dataset_size):
    """
    Print the progress of the training for each epoch
    """
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()


def train_dataset(_epoch, dataloader, model, loss_function, optimizer,n_classes):
    # IMPORTANT: switch to train mode
    # enable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0

    # obtain the model's device ID
    device = next(model.parameters()).device

    for index, batch in enumerate(dataloader, 1):
        # get the inputs (batch)
        inputs, labels, lengths = batch

        # move the batch tensors to the right device        
        inputs = inputs.to(device)

        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each batch!
        optimizer.zero_grad()


        # Step 2 - forward pass: y' = model(x)
        output = model(inputs,lengths)
        
        #import ipdb; ipdb.set_trace()

        # Step 3 - compute loss: L = loss_function(y, y')
        
        # fix label type for different loss func
        if n_classes == 2:
            labels = torch.nn.functional.one_hot(labels, num_classes= 2)
            labels = labels.float()
            
        else:
            labels = labels.long()

        loss = loss_function(output,labels)
        # output :: BS (x 1)       , for bin classification
        #           BS x N_CLASSES , else 
        
        # Step 4 - backward pass: compute gradient wrt model parameters
        loss.backward()

        # Step 5 - update weights
        optimizer.step()


        running_loss += loss.data.item()

        # print statistics
        progress(loss=loss.data.item(),
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))
                 
    print(f' Epoch {_epoch}, Total loss: {running_loss/index:.4f}\n')
    return running_loss / index


def eval_dataset(dataloader, model, loss_function,n_classes):

    # IMPORTANT: switch to eval mode
    # disable regularization layers, such as Dropout
    model.eval()
    running_loss = 0.0

    y_pred = []  # the predicted labels
    y = []  # the gold labels

    # obtain the model's device ID
    device = next(model.parameters()).device

   
    # IMPORTANT: in evaluation mode, we don't want to keep the gradients
    # so we do everything under torch.no_grad()
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            
            # get the inputs (batch)
            inputs, labels, lengths = batch
            
            # Step 1 - move the batch tensors to the right device
            inputs = inputs.to(device)
            
            # Step 2 - forward pass: y' = model(x)
            output = model(inputs,lengths)


            # Step 3 - compute loss.
            
            # We compute the loss only for inspection (compare train/test loss)
            # because we do not actually backpropagate in test time

            # Step 4 - make predictions (class = argmax of posteriors)

            # output :: BS (x 1)       , for bin classification
            #           BS x N_CLASSES , else  
             
            # collect gold labels 
            y += labels.tolist()
            if n_classes == 2:

                # fix label type for different loss func
                labels = torch.nn.functional.one_hot(labels, num_classes= 2)
                labels = labels.float()
            
            else:
                # fix label    
                labels = labels.long()

            # get max class index
            pred = torch.argmax(output,axis = 1).tolist()
            loss = loss_function(output,labels)
                

            # Step 5 - collect the predictions and batch loss
            y_pred += pred
            
            running_loss += loss.data.item()

    return running_loss / index, (y, y_pred)
