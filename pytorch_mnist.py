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
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            # W&B - logging of train loss
            wandb.log({"Train Loss": loss})

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
            
            # W&B - media
            example_images.append(
                wandb.Image(data[0],caption="Pred: {}  |  Truth: {}".format(pred[0].item(),target[0])))

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    # W&B - logging of media, test acc, test loss
    wandb.log({
        "Examples": example_images,
        "Test Accuracy": test_accuracy,
        "Test Loss": test_loss})

# Driver function
def main():

    # W&B hyperparameters to be logged
    hyperparam_defaults = dict(
            batch_size = 200,          
            test_batch_size = 1000,               
            lr = 0.15,               
            momentum = 0.5,
            epochs = 5,                     
            log_interval = 5,
            seed = 42
        )
    # W&B - Initialize a new run and set config
    run = wandb.init(config=hyperparam_defaults, project="pytorch-mnist", save_code=True)
    config = wandb.config

    # Cuda (if applicable)
    use_cuda =  torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Random seeds
    random.seed(config.seed)       
    torch.manual_seed(config.seed) 
    numpy.random.seed(config.seed) 
    torch.backends.cudnn.deterministic = True

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
    
    # W&B - wandb.watch() fetches layer dimensions, gradients, model parameters 
    wandb.watch(model, log="all")

    for epoch in range(1, config.epochs + 1):
        train(config, model, device, train_loader, optimizer, epoch)
        test(config, model, device, test_loader)
        
    # W&B - Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.
    torch.save(model.state_dict(), 'model.onnx')
    wandb.save('model.onnx')

    # W&B - Artifact
    model_artifact = wandb.Artifact('{}-pytorch-mnist-model'.format(run.id), type='model')
    model_artifact.add_file('model.onnx')
    run.log_artifact(model_artifact)
     
if __name__ == '__main__':
    main()
