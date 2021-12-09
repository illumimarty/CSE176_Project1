import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def train_nn(model, criterion, optimizer, trainloader):
    running_loss = 0.0
    correct = 0
    model.train()
    for data, labels in tqdm(trainloader):
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        # Clear the gradients
        optimizer.zero_grad()
        #Forward Pass
        target = model(data)
        # Find the Loss
        loss = criterion(target,labels)
        # Calculate gradients
        loss.backward() 
        # Update Weights
        optimizer.step()
        # Calculate Loss, number of misclassified images by the model
        running_loss += loss.item() * data.size(0)

        # Calculate number of correct classifications
        pred = target.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    
    train_loss = running_loss / len(trainloader.dataset)
    train_accuracy = 100.0 * correct / len(trainloader.dataset)

    return train_loss, train_accuracy

def validate_nn(model, criterion, validloader):
    running_loss = 0.0
    correct = 0
    model.eval()     # Optional when not using Model Specific layer
    for data, labels in validloader:
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        # Forward Pass
        target = model(data)
        # Find the Loss: The number of incorrectly predicted labels
        loss = criterion(target,labels)
        # Calculate Loss
        running_loss += loss.item() * data.size(0)
        # Calculate number of correct classifications
        pred = target.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    
    valid_loss = running_loss / len(validloader.dataset)
    valid_accuracy = 100.0 * correct / len(validloader.dataset)

    return valid_loss, valid_accuracy

def update_optimal_model(model, min_valid_loss, valid_loss, path):
    if min_valid_loss > valid_loss:
        print(f'\n\t\t ------------------------------------------------------------------------------------------- \n')
        print(f'\t\t\t\t\t Observed Errors Decreased({min_valid_loss:.3f}--->{valid_loss:.3f}) \n\t\t\t\t\t Saving The Model')
        min_valid_loss = valid_loss
        
        # Saving State Dict
        torch.save(model.state_dict(), path)
        return min_valid_loss
    else:
        print(f'\n\t\t ------------------------------------------------------------------------------------------- \n')
        print(f'\t\t\t\t\t No decrease in observed errors')
        return min_valid_loss

def process(version, model, trainloader, validloader, epochs, min_valid_loss):
    print("Checking GPU availability...")
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model loaded onto GPU")
    else:
        print("     GPU Not available")

    print("Initialize criterion")
    criterion = nn.CrossEntropyLoss()
    print("Initialize optimizer")
    # lr = Learning Rate
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    if version == "PixelNet":
        path = "PixelNet.pth"
        writer = SummaryWriter("runs/PixelNet")
    elif version == "LeNet5":
        path = "LeNet5.pth"
        writer = SummaryWriter("runs/LeNet5")
    print("Begin training...")

    for e in range(epochs):
        # Train the model and compute the training loss
        train_loss, train_accuracy = train_nn(model, criterion, optimizer, trainloader)
        print(f'Epoch {e+1} \t\t Training Loss: {train_loss} \t\t Train Accuracy: {train_accuracy}')
        writer.add_scalar('Training Loss', train_loss, e + 1)
        writer.add_scalar('Training Accuracy', train_accuracy, e + 1)

        # Compute the loss on the validation set
        valid_loss, valid_accuracy = validate_nn(model, criterion, validloader)
        print(f'\t\t\t Validation Loss: {valid_loss} \t\t Validation Accuracy: {valid_accuracy}')
        writer.add_scalar('Validation Loss', valid_loss, e + 1)
        writer.add_scalar('Validation Accuracy', valid_accuracy, e + 1)

        
        # Check to see if the optimal model can be updated
        min_loss = update_optimal_model(model, min_valid_loss, valid_loss, path)
        min_valid_loss = min_loss
    
    writer.close()