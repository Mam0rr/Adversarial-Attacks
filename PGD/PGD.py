import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='PyTorch MNIST Training with Adversarial Attack')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='Batch size (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='Test batch size (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='Learning rate')
parser.add_argument('--epsilon', type=float, default=0.1, metavar='E',
                    help='Epsilon for attack (default: 0.1)')
parser.add_argument('--alpha', type=float, default=0.01, metavar='A',
                    help='Step size for PGD attack (default: 0.01)')
parser.add_argument('--num-iter', type=int, default=40, metavar='N',
                    help='Number of iterations for PGD (default: 40)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='Random seed')

args = parser.parse_args(args=[])  # Running in Jupyter, so passing args=[]

# Device setup
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Load data
train_set = torchvision.datasets.MNIST(root='data', train=True, download=True,
                                       transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

test_set = torchvision.datasets.MNIST(root='data', train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))
test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

# Neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output

def pgd_attack(model, x, y, epsilon=0.1, alpha=0.01, num_iter=40):

    # Clone the input tensor and initialize adversarial example
    x_adv = x.clone().detach().to(x.device)
    original_x = x.clone().detach().to(x.device)  # Keep original inputs for clamping

    for _ in range(num_iter):
        # Ensure x_adv requires gradients for backpropagation
        x_adv.requires_grad_(True)

        # Forward pass and compute loss
        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()

        # Update adversarial example
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()  # Gradient ascent step
            x_adv = torch.clamp(x_adv, original_x - epsilon, original_x + epsilon)  # Clamp within L-infinity ball
            x_adv = torch.clamp(x_adv, 0, 1)  # Ensure valid pixel range [0, 1]

        # Detach to prevent accumulating gradients in subsequent iterations
        x_adv = x_adv.detach()

    return x_adv

# Training function
def train(args, model, device, train_loader, optimizer):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28 * 28)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), target)
        loss.backward()
        optimizer.step()

# Evaluation function
def eval_test(model, device, loader, attack=None):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28 * 28)
        if attack:
            data = attack(model, data, target, epsilon=args.epsilon, alpha=args.alpha, num_iter=args.num_iter)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loader.dataset)
    test_accuracy = correct / len(loader.dataset)
    return test_loss, test_accuracy

# Main training loop
def train_model():
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Lists to store the accuracy values for plotting
    train_accuracies = []
    adv_accuracies = []

    print(f"{'Epoch':^6} | {'Time (s)':^8} | {'Train Loss':^12} | {'Train Acc (%)':^14} | {'Adv Loss':^12} | {'Adv Acc (%)':^14}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train(args, model, device, train_loader, optimizer)
        trn_loss, trn_acc = eval_test(model, device, train_loader)
        adv_loss, adv_acc = eval_test(model, device, train_loader, attack=pgd_attack)

        # Collect accuracy values
        train_accuracies.append(trn_acc * 100)  # Convert to percentage
        adv_accuracies.append(adv_acc * 100)    # Convert to percentage

        print(f"{epoch:^6} | {int(time.time() - start_time):^8} | {trn_loss:^12.4f} | {trn_acc * 100:^14.2f} | {adv_loss:^12.4f} | {adv_acc * 100:^14.2f}")

    adv_tst_loss, adv_tst_acc = eval_test(model, device, test_loader, attack=pgd_attack)
    print("\nTraining complete!\n")
    print(f"Final Results on Adversarial Test Set: Loss = {adv_tst_loss:.4f}, Accuracy = {adv_tst_acc * 100:.2f}%")
    torch.save(model.state_dict(), 'mnist_pgd_model.pt')

    # Plotting the accuracy graphs
    plot_accuracy(train_accuracies, adv_accuracies)

    return model

# Function to plot the training vs adversarial accuracy
def plot_accuracy(train_accuracies, adv_accuracies):
    epochs = range(1, len(train_accuracies) + 1)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs, adv_accuracies, label='Adversarial Accuracy', marker='x')
    
    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training vs Adversarial Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

# Run model
if __name__ == "__main__":
    model = train_model()
