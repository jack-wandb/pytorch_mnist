
from __future__ import print_function
import random 
import numpy 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# W&B - Import the wandb library
import wandb

# Model architecture definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

# Define training function
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):   
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

# Define test function
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # W&B - Creation of a list of wandb Image objects consisting of images, predicted values, and their ground truth values
            example_images.append(
                wandb.Image(
                data[0], 
                    caption="Pred: {}  |  Truth: {}".format(pred[0].item(),target[0])
                )
            )   
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    # W&B - logging of first 10 example images per test step, along with pred vs GT, test accuracy, test loss
    wandb.log({
        "Examples": example_images,
        "Test Accuracy": test_accuracy,
        "Test Loss": test_loss})

# Driver function
def main():
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    random.seed(config.seed)       
    torch.manual_seed(config.seed) 
    numpy.random.seed(config.seed) 
    torch.backends.cudnn.deterministic = True

    # W&B - Initialize a new run
    run = wandb.init(project="pytorch-mnist", save_code=True)

    # W&B - Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    config.batch_size = 200          
    config.test_batch_size = 1000  
    config.epochs = 5             
    config.lr = 0.15               
    config.momentum = 0.5           
    config.no_cuda = False         
    config.seed = 42               
    config.log_interval = 10
    
    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=config.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=config.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.lr,
                          momentum=config.momentum)
    
    # W&B - wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    wandb.watch(model, log="all")

    for epoch in range(1, config.epochs + 1):
        train(config, model, device, train_loader, optimizer, epoch)
        test(config, model, device, test_loader)
        
        # W&B - Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.
        torch.save(model.state_dict(), 'model.onnx')
        wandb.save('model.onnx')
    
        # W&B Artifacts - Save each model as a W&B Artifact to maintain data lineage and centralize storage of models produced
        # by this run.
        model_artifact = wandb.Artifact('{}-pytorch-mnist-model'.format(run.id), type='model')

        # Add model file to artifact's contents
        model_artifact.add_file('model.onnx')

        # Save artifact version to W&B. W&B will automatically version each of these models for you.
        run.log_artifact(model_artifact)
     
if __name__ == '__main__':
    main()
