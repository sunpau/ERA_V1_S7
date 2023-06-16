import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import GetCorrectPredCount
from tqdm import tqdm


# #------------- Model 1-------------
# #Skeleton formed
# class Net(nn.Module):
  # def __init__(self):
    # super(Net, self).__init__()
    # self.conv1 = nn.Sequential(
        # nn.Conv2d(1, 16, 3, bias=False, padding=1), #Image Input: 28x28x1 -> 28x28x16  #Receptive Field 1 -> 3
        # nn.ReLU(),
        # nn.Conv2d(16, 32, 3, bias=False), #Input: 28x28x16 -> 26x26x32  #Receptive Field 3 -> 5
        # nn.ReLU(),
        # #Transition Block = MaxPool + 1x1 Convolution
        # #nn.Conv2d(16, 8, 1, bias=False),
        # nn.MaxPool2d(2, 2),    #Input: 13x13x32 -> 13x13x32  #Receptive Field  5 -> 6
        # nn.Conv2d(32, 16, 1, bias=False)   #Input: 13x13x32 -> 13x13x16  #Receptive Field  6 -> 6
    # )

    # self.conv2 = nn.Sequential(
        # nn.Conv2d(16, 16, 3, bias=False),  #Input: 13x13x16 -> 11x11x16  #Receptive Field  6 -> 10
        # nn.ReLU(),
        # nn.Conv2d(16, 32, 3, bias=False),  #Input: 11x11x16 -> 9x9x32  #Receptive Field  10 -> 14
        # nn.ReLU()
    # )
    # self.conv3 = nn.Sequential(
        # nn.Conv2d(32, 16, 3, bias=False),  #Input: 9x9x32 -> 7x7x16  #Receptive Field  14 -> 18
        # nn.ReLU(),
        # nn.Conv2d(16, 16, 3, bias=False),  #Input: 7x7x16 -> 5x5x16  #Receptive Field  18 -> 22
        # nn.ReLU()
    # )
    # self.conv4 = nn.Sequential(
        # nn.Conv2d(16, 10, kernel_size=(5, 5), bias=False) #Input: 5x5x16 -> 1x1x10  #Receptive Field  22 -> 30
        # )

  # def forward(self, x):
    # x = self.conv1(x)
    # x = self.conv2(x)
    # x = self.conv3(x)
    # x = self.conv4(x)
    # x = x.view(-1, 10)
    # x = F.log_softmax(x, dim=1)
    # return x
	
	
# #------------- Model 2-------------
# #Lighter Model
# class Net(nn.Module):
  # def __init__(self):
    # super(Net, self).__init__()
    # self.conv1 = nn.Sequential(
        # nn.Conv2d(1, 8, 3, bias=False, padding=1), #Image Input: 28x28x1 -> 28x28x16  #Receptive Field 1 -> 3
        # nn.ReLU(),
        # nn.Conv2d(8, 16, 3, bias=False), #Input: 28x28x16 -> 26x26x32  #Receptive Field 3 -> 5
        # nn.ReLU(),
        # #Transition Block = MaxPool + 1x1 Convolution
        # #nn.Conv2d(16, 8, 1, bias=False),
        # nn.MaxPool2d(2, 2),    #Input: 13x13x32 -> 13x13x32  #Receptive Field  5 -> 6
        # nn.Conv2d(16, 8, 1, bias=False)   #Input: 13x13x32 -> 13x13x16  #Receptive Field  6 -> 6
    # )

    # self.conv2 = nn.Sequential(
        # nn.Conv2d(8, 8, 3, bias=False),  #Input: 13x13x16 -> 11x11x16  #Receptive Field  6 -> 10
        # nn.ReLU(),
        # nn.Conv2d(8, 8, 3, bias=False),  #Input: 11x11x16 -> 9x9x32  #Receptive Field  10 -> 14
        # nn.ReLU()
    # )
    # self.conv3 = nn.Sequential(
        # nn.Conv2d(8, 16, 3, bias=False),  #Input: 9x9x32 -> 7x7x16  #Receptive Field  14 -> 18
        # nn.ReLU(),
        # nn.Conv2d(16, 16, 3, bias=False),  #Input: 7x7x16 -> 5x5x16  #Receptive Field  18 -> 22
        # nn.ReLU()
    # )
    # self.conv4 = nn.Sequential(
        # nn.Conv2d(16, 10, kernel_size=(5, 5), bias=False) #Input: 5x5x16 -> 1x1x10  #Receptive Field  22 -> 30
        # )

  # def forward(self, x):
    # x = self.conv1(x)
    # x = self.conv2(x)
    # x = self.conv3(x)
    # x = self.conv4(x)
    # x = x.view(-1, 10)
    # x = F.log_softmax(x, dim=1)
    # return x

# #------------- Model 3-------------	
# #Batch Norm Added
# class Net(nn.Module):
  # def __init__(self):
    # super(Net, self).__init__()
    # self.conv1 = nn.Sequential(
        # nn.Conv2d(1, 8, 3, bias=False, padding=1), #Image Input: 28x28x1 -> 28x28x16  #Receptive Field 1 -> 3
        # nn.ReLU(),
        # nn.BatchNorm2d(8),
        # nn.Conv2d(8, 16, 3, bias=False), #Input: 28x28x16 -> 26x26x32  #Receptive Field 3 -> 5
        # nn.ReLU(),
        # nn.BatchNorm2d(16),
        # #Transition Block = MaxPool + 1x1 Convolution
        # #nn.Conv2d(16, 8, 1, bias=False),
        # nn.MaxPool2d(2, 2),    #Input: 13x13x32 -> 13x13x32  #Receptive Field  5 -> 6
        # nn.Conv2d(16, 8, 1, bias=False)   #Input: 13x13x32 -> 13x13x16  #Receptive Field  6 -> 6
    # )

    # self.conv2 = nn.Sequential(
        # nn.Conv2d(8, 8, 3, bias=False),  #Input: 13x13x16 -> 11x11x16  #Receptive Field  6 -> 10
        # nn.ReLU(),
        # nn.BatchNorm2d(8),
        # nn.Conv2d(8, 8, 3, bias=False),  #Input: 11x11x16 -> 9x9x32  #Receptive Field  10 -> 14
        # nn.ReLU(),
        # nn.BatchNorm2d(8)
    # )
    # self.conv3 = nn.Sequential(
        # nn.Conv2d(8, 16, 3, bias=False),  #Input: 9x9x32 -> 7x7x16  #Receptive Field  14 -> 18
        # nn.ReLU(),
        # nn.BatchNorm2d(16),
        # nn.Conv2d(16, 16, 3, bias=False),  #Input: 7x7x16 -> 5x5x16  #Receptive Field  18 -> 22
        # nn.ReLU(),
        # nn.BatchNorm2d(16)
    # )
    # self.conv4 = nn.Sequential(
        # nn.Conv2d(16, 10, kernel_size=(5, 5), bias=False) #Input: 5x5x16 -> 1x1x10  #Receptive Field  22 -> 30
        # )

  # def forward(self, x):
    # x = self.conv1(x)
    # x = self.conv2(x)
    # x = self.conv3(x)
    # x = self.conv4(x)
    # x = x.view(-1, 10)
    # x = F.log_softmax(x, dim=1)
    # return x
	
# #------------- Model 4-------------		
# #Added Dropout
# class Net(nn.Module):
  # def __init__(self):
    # super(Net, self).__init__()
    # dropout = 0.1
    # self.conv1 = nn.Sequential(
        # nn.Conv2d(1, 8, 3, bias=False, padding=1), #Image Input: 28x28x1 -> 28x28x16  #Receptive Field 1 -> 3
        # nn.ReLU(),
        # nn.BatchNorm2d(8),
        # nn.Dropout(dropout),
        # nn.Conv2d(8, 16, 3, bias=False), #Input: 28x28x16 -> 26x26x32  #Receptive Field 3 -> 5
        # nn.ReLU(),
        # nn.BatchNorm2d(16),
        # nn.Dropout(dropout),
        # #Transition Block = MaxPool + 1x1 Convolution
        # #nn.Conv2d(16, 8, 1, bias=False),
        # nn.MaxPool2d(2, 2),    #Input: 13x13x32 -> 13x13x32  #Receptive Field  5 -> 6
        # nn.Conv2d(16, 8, 1, bias=False)   #Input: 13x13x32 -> 13x13x16  #Receptive Field  6 -> 6
    # )

    # self.conv2 = nn.Sequential(
        # nn.Conv2d(8, 8, 3, bias=False),  #Input: 13x13x16 -> 11x11x16  #Receptive Field  6 -> 10
        # nn.ReLU(),
        # nn.BatchNorm2d(8),
        # nn.Dropout(dropout),
        # nn.Conv2d(8, 8, 3, bias=False),  #Input: 11x11x16 -> 9x9x32  #Receptive Field  10 -> 14
        # nn.ReLU(),
        # nn.BatchNorm2d(8),
        # nn.Dropout(dropout)
    # )
    # self.conv3 = nn.Sequential(
        # nn.Conv2d(8, 16, 3, bias=False),  #Input: 9x9x32 -> 7x7x16  #Receptive Field  14 -> 18
        # nn.ReLU(),
        # nn.BatchNorm2d(16),
        # nn.Dropout(dropout),
        # nn.Conv2d(16, 16, 3, bias=False),  #Input: 7x7x16 -> 5x5x16  #Receptive Field  18 -> 22
        # nn.ReLU(),
        # nn.BatchNorm2d(16)
    # )
    # self.conv4 = nn.Sequential(
        # nn.Conv2d(16, 10, kernel_size=(5, 5), bias=False) #Input: 5x5x16 -> 1x1x10  #Receptive Field  22 -> 30
        # )

  # def forward(self, x):
    # x = self.conv1(x)
    # x = self.conv2(x)
    # x = self.conv3(x)
    # x = self.conv4(x)
    # x = x.view(-1, 10)
    # x = F.log_softmax(x, dim=1)
    # return x

# #------------- Model 5-------------
# #GAP replaced heavy kernal
# class Net(nn.Module):
  # def __init__(self):
    # super(Net, self).__init__()
    # dropout = 0.1
    # self.conv1 = nn.Sequential(
        # nn.Conv2d(1, 8, 3, bias=False, padding=1), #Image Input: 28x28x1 -> 28x28x16  #Receptive Field 1 -> 3
        # nn.ReLU(),
        # nn.BatchNorm2d(8),
        # nn.Dropout(dropout),
        # nn.Conv2d(8, 16, 3, bias=False), #Input: 28x28x16 -> 26x26x32  #Receptive Field 3 -> 5
        # nn.ReLU(),
        # nn.BatchNorm2d(16),
        # nn.Dropout(dropout),
        # #Transition Block = MaxPool + 1x1 Convolution
        # #nn.Conv2d(16, 8, 1, bias=False),
        # nn.MaxPool2d(2, 2),    #Input: 13x13x32 -> 13x13x32  #Receptive Field  5 -> 6
        # nn.Conv2d(16, 8, 1, bias=False)   #Input: 13x13x32 -> 13x13x16  #Receptive Field  6 -> 6
    # )

    # self.conv2 = nn.Sequential(
        # nn.Conv2d(8, 8, 3, bias=False),  #Input: 13x13x16 -> 11x11x16  #Receptive Field  6 -> 10
        # nn.ReLU(),
        # nn.BatchNorm2d(8),
        # nn.Dropout(dropout),
        # nn.Conv2d(8, 8, 3, bias=False),  #Input: 11x11x16 -> 9x9x32  #Receptive Field  10 -> 14
        # nn.ReLU(),
        # nn.BatchNorm2d(8),
        # nn.Dropout(dropout)
    # )
    # self.conv3 = nn.Sequential(
        # nn.Conv2d(8, 16, 3, bias=False),  #Input: 9x9x32 -> 7x7x16  #Receptive Field  14 -> 18
        # nn.ReLU(),
        # nn.BatchNorm2d(16),
        # nn.Dropout(dropout),
        # nn.Conv2d(16, 10, 3, bias=False),  #Input: 7x7x16 -> 5x5x16  #Receptive Field  18 -> 22
        # nn.ReLU(),
        # nn.BatchNorm2d(10)
    # )
    # self.conv4 = nn.Sequential(
        # nn.AdaptiveAvgPool2d((1, 1))
        # )

  # def forward(self, x):
    # x = self.conv1(x)
    # x = self.conv2(x)
    # x = self.conv3(x)
    # x = self.conv4(x)
    # x = x.view(-1, 10)
    # x = F.log_softmax(x, dim=1)
    # return x
	
#------------- Model 6-------------	
#Extra Layer added before GAP
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    dropout = 0.1
    self.conv1 = nn.Sequential(
        nn.Conv2d(1, 8, 3, bias=False, padding=1), #Image Input: 28x28x1 -> 28x28x16  #Receptive Field 1 -> 3
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Dropout(dropout),
        nn.Conv2d(8, 16, 3, bias=False), #Input: 28x28x16 -> 26x26x32  #Receptive Field 3 -> 5
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Dropout(dropout),
        #Transition Block = MaxPool + 1x1 Convolution
        #nn.Conv2d(16, 8, 1, bias=False),
        nn.MaxPool2d(2, 2),    #Input: 13x13x32 -> 13x13x32  #Receptive Field  5 -> 6
        nn.Conv2d(16, 8, 1, bias=False)   #Input: 13x13x32 -> 13x13x16  #Receptive Field  6 -> 6
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(8, 8, 3, bias=False),  #Input: 13x13x16 -> 11x11x16  #Receptive Field  6 -> 10
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Dropout(dropout),
        nn.Conv2d(8, 8, 3, bias=False),  #Input: 11x11x16 -> 9x9x32  #Receptive Field  10 -> 14
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Dropout(dropout)
    )
    self.conv3 = nn.Sequential(
        nn.Conv2d(8, 16, 3, bias=False),  #Input: 9x9x32 -> 7x7x16  #Receptive Field  14 -> 18
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Dropout(dropout),
        nn.Conv2d(16, 10, 3, bias=False),  #Input: 7x7x16 -> 5x5x16  #Receptive Field  18 -> 22
        nn.ReLU(),
        nn.BatchNorm2d(10)
    )
    self.conv4 = nn.Sequential(
        nn.Conv2d(10, 10,3, bias=False), #Input: 5x5x16 -> 1x1x10  #Receptive Field  22 -> 26
        nn.AdaptiveAvgPool2d((1, 1))
        )

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = x.view(-1, 10)
    x = F.log_softmax(x, dim=1)
    return x
    
def train_model(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()  # zero the gradients- not to use perious gradients

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()   #updates the parameter - gradient descent
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
  train_acc = 100*correct/processed
  train_loss = train_loss/len(train_loader)
  return train_acc, train_loss
  

def test_model(model, device, test_loader, criterion):
    model.eval() #set model in test (inference) mode

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_acc, test_loss
# def model_summary():
# 	!pip install torchsummary
# 	from torchsummary import summary
# 	use_cuda = torch.cuda.is_available()
# 	device = torch.device("cuda" if use_cuda else "cpu")
# 	model = Net().to(device)
# 	summary(model, input_size=(1, 28, 28))