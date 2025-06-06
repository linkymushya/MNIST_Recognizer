import torch
from torch import nn, optim
from data_preprocess import data_loader_MNIST


train_loader, test_loader = data_loader_MNIST()


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 526),
            nn.ReLU(),
            nn.Linear(526, 126),
            nn.ReLU(),
            nn.Linear(126, 10)
        )

    def forward(self, x):
        return self.layers(x)


model = MyModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


total_train_step = 0
total_test_step = 0


num_epochs = 10
for epoch in range(num_epochs):
    print("-----第{}次训练开始-----".format(epoch+1))

    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == labels).sum()
            total_accuracy = total_accuracy + accuracy
        print("测试集上的loss：{}，训练集上的正确率：{}".format(total_test_loss / len(test_loader.dataset), total_accuracy / len(test_loader.dataset)))
torch.save(model.state_dict(), 'mnist_mlp.pth')
