import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn

training_data = datasets.MNIST(root=".", train=True, download=True, transform=ToTensor()) 

test_data = datasets.MNIST(root=".", train=False, download=True, transform=ToTensor())

f = open("ai_lr1.txt", "w")
g = open("accuracy_lr1.txt", "w")

#datasets.(Dataset 이름)(root = (파일 저장 경로), train = bool, download = True, transform = ToTensor())
#torch가 알아먹을 수 있도록 tensor로 만드는듯
#바탕화면 폴더에 지금 MNIST가 저장되어 있음.

figure = plt.figure(figsize=(8, 8))
cols, rows = 5, 5

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

#matplotlib를 이용해 데이터의 일부를 보여주는 부분이나 필요없으니 생략

loaded_train = DataLoader(training_data, batch_size=64, shuffle=True)
loaded_test = DataLoader(test_data, batch_size=64, shuffle=True)

#미니 batch로 샘플 전달, shuffle을 해서 과적합을 막는다.

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        #1차원 텐서로 변환 -> 인공신경망에 입력값으로 제공할 수 있게 함. (왜 2차원 이상은 입력이 안 될까?)
        #self.flatten은 함수이다.

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        #nn.Linear(): 선형 회귀 모델 (왜 Linear을 쓰는 걸까?)
        #nn.ReLU(): Activation Function, 비선형, relu(x) = max(x, 0)
        #nn.Sequential(): 순차적으로 모듈을 실행, 신경망 깊이 깊어질수록 편리함. -> 데이터를 다양한 방식으로 변환.
        #지금 정의된 self.linear_relu_stack도 함수이다!

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits #self.linear_relu.stack으로 데이터를 변환한 값을 리턴함.

model = NeuralNetwork()
#print(model)

# 출력 결과
# NeuralNetwork(
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (linear_relu_stack): Sequential(
#     (0): Linear(in_features=784, out_features=512, bias=True)
#     (1): ReLU() -> 
#     (2): Linear(in_features=512, out_features=512, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=512, out_features=10, bias=True)
#   )
# )

loss_function = nn.CrossEntropyLoss()
#오차 함수

optimizer = torch.optim.SGD(model.parameters(), lr=1)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch:
            loss, current = loss.item(), batch * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            f.write(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}")
    g.write(f"Accuracy: {(100*correct):>0.1f}\n")

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    f.write(f"Epoch {t+1}\n-------------------------------\n")
    train(loaded_train, model, loss_function, optimizer)
    test(loaded_test, model, loss_function)
print("Done!")
f.write("Done!")
f.close()
g.close()