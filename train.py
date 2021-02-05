import torch
import torch.nn as nn
from torchvision import datasets, transforms

from net import Model

# hyperparameters
EPOCHS = 10
LEARNING_RATE = 1.
GAMMA = 0.7
BATCH_SIZE = 64
TEST_BATCH_SIZE = 500

def train(model, data, optimizer, device):
    """ This function does one epoch of training """
    model.train()
    for samples, labels in data:
        samples, labels = samples.to(device), labels.to(device)
        optimizer.zero_grad()
        results = model(samples)
        loss = nn.functional.nll_loss(results, labels)
        loss.backward()

        optimizer.step()

def test(model, data, device):
    """ This function evaluates the network on the test set """
    model.eval()
    ctr = 0
    with torch.no_grad():
        for samples, labels in data:
            samples, labels = samples.to(device), labels.to(device)
            results = model(samples)
            numbers = results.argmax(dim=1, keepdim=True)
            ctr += int(numbers.eq(labels.view_as(numbers)).sum())
    acc = ctr/len(data.dataset)
    print('Trained. Accuracy on test set - %.5f'%acc)

def main():
    device = torch.device('cpu')
    transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]) # MNIST-specific values
    train_set = datasets.MNIST('mnist_data', train=True, transform=transform)
    test_set = datasets.MNIST('mnist_data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=TEST_BATCH_SIZE)

    model = Model()
    model = model.to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=GAMMA, step_size=1)
    for i in range(EPOCHS):
        train(model, train_loader, optimizer, device)
        scheduler.step()

    test(model, test_loader, device)
    torch.save(model.state_dict(), "parameters.pt")

if __name__ == '__main__':
    main()
