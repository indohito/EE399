# EE399HW4

## Akash Shetty, HW4

## Introduction and Overview
This is our 4th homework of our machine learning class, in this homework we will be building neural networks to analyze simple data and the MNIST dataset, we will be implementing a Feed Forward Neural Network and Long Short Term Memory. We will be using Pytorch to build these networks and Stochastic Gradient Descent as the optimizer. 

## Theoretical Backround
### Stochastic Gradient Descent 
Stochastic Gradient Descent is a optimization algorithm that is commonly used in machine learning, it works using the gradient descent algorithms however the differences come where SGD selects mini batches of the data and computes the negative gradient among them for a number of epochs. Below is a Visualization of this technique. 

<img width="300" alt="image" src="https://github.com/indohito/EE399/assets/107958888/f024ad17-ff71-4f0e-add3-aeec02deb406">

### Feed Forward Neural Netowrk 
A Feed Forward Neural Network is also known as a multilayer perceptron the networks by the data flows one direction from the input to the output. These networks usually consist of layers with neurons of different weights connecting to the other layers, usually a input layer some number of hidden layers, and an output layer. The hidden layers applies nonlinear transformations to the data allowing it to analyze complex patterns. These networks usually require high levels of hyparameter tuning. 

<img width="300" alt="image" src="https://github.com/indohito/EE399/assets/107958888/c271102d-6e7d-4239-8b42-49a4dae066b5">

### Long Short Term Memory
Long Short Term Memory is a type of Recurrent Neural Network, it is meant to help the vanishing gradient problem which is where the gradient become very small during back prop which makes the network harder to learn long term data. LSTM is a way to fix this problem by creating a memory cell where they take the information cells and these cells steps are retained with a forget, input, and output parts of the cell. Below is a visualization of this 

<img width="300" alt="image" src="https://github.com/indohito/EE399/assets/107958888/16433e00-1e94-4a01-bed7-37b19f29985b">

## Algorithm Implementation and Computational Results

### Part 1
#### Fit the data to a three layer feed forward neural network. Using the first 20 data points as training data, fit the neural network. Compute the least-square error for each of these over the training points. Then compute the least square error of these models on the test data which are the remaining 10 data points.
```python
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
              40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50,1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
After fitting the model We go these results from using the first 20 points as training data and the last 10 as test data we get these results:
<img width="300" alt="image" src="https://github.com/indohito/EE399/assets/107958888/942abd62-2415-4798-8d47-a874abe8191e">  

And the Testing data  gives us these results

<img width="190" alt="image" src="https://github.com/indohito/EE399/assets/107958888/318be95a-b1fb-4855-97bf-a5bbb163d861">

#### Repeat but use the first 10 and last 10 data points as training data. Then fit the model to the test data (which are the 10 held out middle data points). Compare these results to the last one 
After fitting the model with this data we get these results for training data 

<img width="279" alt="image" src="https://github.com/indohito/EE399/assets/107958888/f0e0b39e-e82e-4499-9053-ed7de10a4498">

The testing data gives us these results 

<img width="194" alt="image" src="https://github.com/indohito/EE399/assets/107958888/1f7487a7-3af3-4ed8-b621-0e53b36f41bd">

The fit in these models show more overfitting in the first data set where it was the first 20 because the loss on the testing results is much higher

#### Compare the models fit in homework one to the neural networks
The models in these nerual networks were able to perform only better than the polynomial data fit in homework one but the results for linear and parabolic fit were better most likely because these didnt overfit the data. 

### PART 2
#### Now train a feedforward neural network on the MNIST data set. You will start by performing the following analysis: Compute the first 20 PCA modes of the digit images.

```python
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

train_images = train_dataset.data.reshape((len(train_dataset), -1))
test_images = test_dataset.data.reshape((len(test_dataset), -1))

pca = PCA(n_components=20)
train_images_pca = pca.fit_transform(train_images)
train_images_pca = pca.transform(test_images)
```

<img width="428" alt="image" src="https://github.com/indohito/EE399/assets/107958888/817d9097-5dba-4c5a-9246-48779ace2f39">


#### Build a feed-forward neural network to classify the digits. Compare the results ofthe neural network against LSTM, SVM (support vector machines) and decision tree classifiers.
FFNN: 
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, 10)
    def forward(self, x):
        x = x.view(-1, 784) # flatten the input image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Train the network
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

<img width="375" alt="image" src="https://github.com/indohito/EE399/assets/107958888/7ec52c86-cbd0-4542-af4e-8f6d5ba1e15a">

LSTM:
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out
```

<img width="359" alt="image" src="https://github.com/indohito/EE399/assets/107958888/872d19ab-5409-4518-9179-ee1eeac89c7e">

In terms of recognizing these digits the LSTM had a higher accuracy percentage as shown below perhaps because adjusting the weights with memory provides a better understanding of the data of the digits. However the FFNN did worse than both the support vector machine in the last homework and the descion tree classifier in the last homework, leading us to believe that a FFFNN may not be the best in classifying these digits unless there is very high levels of data to be used. 
## Summary 
In conclusion I think this homework was a good introduction to the use of neural networks and starting to learn more about them and different optimization techniques. 
