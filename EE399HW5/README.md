# EE399HW5

## Akash Shetty, HW5

## Introduction and Overview
In this homework we will be using the neural networks to forecast the dynamics of the system introdced by the lorenz equations. We want to see how well these neural networks model the chaos that is introduced by these equations
## Theoretical Backround
### Lorenz
The Lorenz Equations are a set of nonlinear differential equations that can be used to describe weather patterns they are known for chaotic and unknown behavior.   

$\frac{{dx}}{{dt}} = \sigma(y - x)$  

$\frac{{dy}}{{dt}} = x(\rho - z) - y$  

$\frac{{dz}}{{dt}} = xy - \beta z$  

### Feed Forward Neural Network 
A Feed Forward Neural Network is also known as a multilayer perceptron the networks by the data flows one direction from the input to the output. These networks usually consist of layers with neurons of different weights connecting to the other layers, usually a input layer some number of hidden layers, and an output layer. The hidden layers applies nonlinear transformations to the data allowing it to analyze complex patterns. These networks usually require high levels of hyparameter tuning. 

<img width="300" alt="image" src="https://github.com/indohito/EE399/assets/107958888/c271102d-6e7d-4239-8b42-49a4dae066b5">

### Long Short Term Memory 
Long Short Term Memory is a type of Recurrent Neural Network, it is meant to help the vanishing gradient problem which is where the gradient become very small during back prop which makes the network harder to learn long term data. LSTM is a way to fix this problem by creating a memory cell where they take the information cells and these cells steps are retained with a forget, input, and output parts of the cell. Below is a visualization of this 

<img width="300" alt="image" src="https://github.com/indohito/EE399/assets/107958888/16433e00-1e94-4a01-bed7-37b19f29985b">

### Recurrent Neural Network
Recurrent Neural Networks are a type of Neural Network that maintain the internal states of the network they use the previous inputs to determine states for the subsequent inputs in the system so that it updates the current states as well. 

<img width="300" alt="image" src="https://github.com/indohito/EE399/assets/107958888/6ed0de61-b242-48f2-9f63-7847e1550009">

### Echo State Neural Network
Echo State Neural Networks are a type of recurrent Neural Networks. They consist of 3 layers the input layer, reservoir layer, and an output layer. the part that is important in an ESNN is the randomly connected neurons with fixed weights that are in the reservoir layer, this random pattern allows the network to capture complex patterns in data. 

<img width="300" alt="image" src="https://github.com/indohito/EE399/assets/107958888/5ecfaae5-3305-465e-a423-dee4eb72e0f6">


## Algorithm Implementation and Computational Results

#### Train a NN to advance the solution from t to t + ∆t for ρ = 10, 28 and 40. Now see how well your NN works for future state prediction for ρ = 17 and ρ = 35.
```python
def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
```
```python 
for rho in [10, 28, 40]:
    # Generate the training data
    x0 = -15 + 30 * np.random.random((100, 3))
    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(sigma, beta, rho))
                      for x0_j in x0])
    nn_input = torch.tensor(x_t[:,:-1,:], dtype=torch.float32)
    nn_output = torch.tensor(x_t[:,1:,:], dtype=torch.float32)
    dataset = TensorDataset(nn_input, nn_output)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train the model
    model = FFNN()
    print(f"Train for rho {rho}")
    train(model, dataloader, criterion, optimizer)
    print(" ")
    print(" ")
# Use the trained model for future state prediction for ρ = 17 and ρ = 35
for rho in [17, 35]:
    x0 = -15 + 30 * np.random.random((100, 3))
    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(sigma, beta, rho))
                      for x0_j in x0])
    nn_input = torch.tensor(x_t[:,:-1,:], dtype=torch.float32)
    nn_output = torch.tensor(x_t[:,1:,:], dtype=torch.float32)
    dataset = TensorDataset(nn_input, nn_output)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            test_loss = criterion(outputs, targets)
        print(f'Test_Loss for {rho}: {test_loss.item():.4f}')
```
#### Compare feed-forward, LSTM, RNN and Echo State Networks for forecasting the dynamics.
```python
class FFNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

<img width="200" alt="image" src="https://github.com/indohito/EE399/assets/107958888/6549b864-d48e-4ac3-8a88-b76d7d5a8ac2">

```python
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=50, num_layers=3)
        self.fc1 = nn.Linear(in_features=50, out_features=3)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc1(x)

        return x
```

<img width="200" alt="image" src="https://github.com/indohito/EE399/assets/107958888/5a42d45c-8b76-43af-a765-7bda752719db">

```python
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=3, hidden_size=60, num_layers=3)
        self.fc1 = nn.Linear(in_features=60, out_features=3)
        
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc1(x)

        return x
```

<img width="200" alt="image" src="https://github.com/indohito/EE399/assets/107958888/2950c828-8ac9-4d49-9bbe-cfc519a0c187">

```python
class ESN(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, reservoir_dim=100, connectivity=.6):
        super().__init__()
        self.reservoir_dim = reservoir_dim
        self.input_to_reservoir = nn.Linear(in_dim, reservoir_dim)
        self.input_to_reservoir.requires_grad_(False)
        self.reservoir = Reservoir(reservoir_dim, connectivity)
        self.readout = nn.Linear(reservoir_dim, out_dim)
  
    def forward(self, x):
        reservoir_in = self.input_to_reservoir(x)
        h = torch.ones(x.size(0), self.reservoir_dim)
        reservoirs = []
        for i in range(x.size(1)):
            out, h = self.reservoir(reservoir_in[:, i, :], h)
            reservoirs.append(out.unsqueeze(1))
        reservoirs = torch.cat(reservoirs, dim=1)
        outputs = self.readout(reservoirs)
        return outputs
```
In these we can see that a LSTM provided the best model for the Lorenz Equations
## Summary 
In this assignment, the task is to explore the prediction capabilities of different neural network architectures for the Lorenz equations. The provided code from class emails is used to train a neural network to advance the solution from time t to t + ∆t for three different values of ρ: 10, 28, and 40. The objective is to assess the performance of the trained neural network in predicting future states for ρ values of 17 and 35.

The first part of the assignment involves training a neural network using the given code and data for the three specified ρ values. The trained neural network is then evaluated by using it to predict future states for two additional ρ values: 17 and 35. The performance of the network in accurately forecasting the dynamics for these unseen ρ values will serve as a measure of its generalization ability.

The second part of the assignment requires comparing different types of neural network architectures for forecasting the dynamics of the Lorenz equations. Specifically, the architectures to be compared are feed-forward networks, Long Short-Term Memory (LSTM) networks, Recurrent Neural Networks (RNNs), and Echo State Networks (ESNs). The aim is to analyze and contrast the predictive capabilities of these different architectures and determine which one performs best for the given task.

By completing this assignment, I gained insights into the strengths and weaknesses of various neural network architectures when applied to time series forecasting tasks, particularly for the Lorenz equations. This exercise enhances understanding of different types of neural networks and their suitability for different prediction tasks, including chaotic and nonlinear systems like the Lorenz equations.
