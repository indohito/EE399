# EE399HW6
# Akash Shetty, HW6
## Introduction and Overview 
This homework will be going over SHRED which are models that learn a mapping from trajectories of sensor measurements to a high-dimensional, spatio-temporal state. The github we pulled from uses dat for sea-surface temperature (SST), a forced turbulent flow, and atmospheric ozone concentration
## Theoretical Backround
### Long Short Term Memory 
Long Short Term Memory is a type of Recurrent Neural Network, it is meant to help the vanishing gradient problem which is where the gradient become very small during back prop which makes the network harder to learn long term data. LSTM is a way to fix this problem by creating a memory cell where they take the information cells and these cells steps are retained with a forget, input, and output parts of the cell. Below is a visualization of this 

<img width="300" alt="image" src="https://github.com/indohito/EE399/assets/107958888/16433e00-1e94-4a01-bed7-37b19f29985b">

### Time lag
In machine learning, time lag refers to the delay or gap between the occurrence of an event and its corresponding effect or observation. It is a concept commonly encountered in time series analysis, where the goal is to predict future values based on historical data. Time lag is important because it allows machine learning models to capture the temporal dependencies and patterns in sequential data. By considering the relationship between past events and their subsequent outcomes, models can make predictions about future events. When working with time series data, it is common to create lag features, which are variables derived from previous observations at different time intervals. These lag features serve as inputs to the machine learning model and help capture the time-dependent relationships in the data.

### Gaussian Noise
In machine learning, Gaussian noise refers to random noise that follows a Gaussian or normal distribution. It is commonly used to add random variations to data during the training phase of a machine learning model. Gaussian noise has several effects and purposes in machine learning. Adding Gaussian noise to the input data can act as a form of regularization. It helps prevent overfitting by introducing randomness to the training samples. By perturbing the data with Gaussian noise, the model is encouraged to learn more robust and generalized patterns, rather than relying too heavily on specific features or noise-free data points.
### Sensors
Changing the number of sensors in machine learning can have various implications depending on the specific application and the data requirements of the machine learning task. Here are a few scenarios to consider. Adding more sensors: Increasing the number of sensors can provide additional data sources, leading to richer and more comprehensive information for the machine learning model. This can potentially improve the model's accuracy, robustness, and ability to capture complex relationships. However, it may also introduce challenges such as increased data dimensionality, data integration complexity, and higher computational requirements for processing the additional sensor data.
## Algorithm Implementation and Computational Results
#### Download the example code (and data) for sea-surface temperature which uses an LSTM/decoder. Train the model and plot the results
Getting datasets to use on LSTM where we can configure num sensors the time lag and the gaussian noise. 
```python 
def get_dataset(sc, device, load_X, num_sensors, lags, gaussian_noise):
    sensor_locations = np.random.choice(m, size=num_sensors, replace=False)
    train_indices = np.random.choice(n - lags, size=1000, replace=False)
    mask = np.ones(n - lags)
    mask[train_indices] = 0
    valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
    valid_indices = valid_test_indices[::2]
    test_indices = valid_test_indices[1::2]
    
    ### create gaussian noise to add to data
    noise = np.random.normal(0, 1, (n, m)) * gaussian_noise
    sc = sc.fit(load_X[train_indices])
    transformed_X = sc.transform(load_X) + noise
    
    ### Generate input sequences to a SHRED model
    all_data_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i+lags, sensor_locations]
        
    ### Generate training validation and test datasets both for reconstruction of states and forecasting sensors

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    ### -1 to have output be at the same time as final sensor measurements
    train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
    
    return train_dataset, valid_dataset, test_dataset
 ```
 #### Do an analysis of the performance as a function of the time lag variable
 The data for the new set of performances
 ```python
 num_sensors = 3 
lags_change = [11, 22, 33, 44, 52]
gaussian_noise = 0
 ``` 
 ```python
lag_charting = []
for l in lags_change:
    train_dataset_lag, valid_dataset_lag, test_dataset_lag = get_dataset(sc, device, load_X, num_sensors, l, gaussian_noise)
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset_lag, valid_dataset_lag, batch_size=64, num_epochs=100, lr=1e-3, verbose=True, patience=5)
    test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    mse_lag = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
    lag_charting.append(mse_lag)
 ```
 
 <img width="478" alt="image" src="https://github.com/indohito/EE399/assets/107958888/26b5bef0-e40e-4728-833b-cc6d9390c835">
 
#### Do an analysis of the performance as a function of noise (add Gaussian noise to data)
The data for performance of noise 
```python
num_sensors = 3
lags = 52
gaussian_noise = [0, .26, .55, .77, 1]
```
```python
gause_charting = []
for g in gaussian_noise:
    train_dataset_gause, valid_dataset_gause, test_dataset_gause = get_dataset(sc, device, load_X, num_sensors, lags, g)
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset_gause, valid_dataset_gause, batch_size=64, num_epochs=100, lr=1e-3, verbose=True, patience=5)
    test_recons = sc.inverse_transform(shred(test_dataset_gause.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset_gause.Y.detach().cpu().numpy())
    mse_lag = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
    gause_charting.append(mse_lag)
    print(" ")
```

<img width="479" alt="image" src="https://github.com/indohito/EE399/assets/107958888/262eae38-7ddf-4788-ba95-331f7c8892b3">

#### Do an analysis of the performance as a function of the number of sensors
The data for performance for the number of sensors 
```python
num_sensors_change = [3, 4, 5, 6, 7]
lags = 52
gaussian_noise = 0
```
```python
sensor_charting = []
for s in num_sensors_change:
    train_dataset_sen, valid_dataset_sen, test_dataset_sen = get_dataset(sc, device, load_X, s, lags, gaussian_noise)
    shred = models.SHRED(s, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset_sen, valid_dataset_sen, batch_size=64, num_epochs=100, lr=1e-3, verbose=True, patience=5)
    test_recons = sc.inverse_transform(shred(test_dataset_sen.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset_sen.Y.detach().cpu().numpy())
    mse_lag = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
    sensor_charting.append(mse_lag)
    print(" ")
```

<img width="497" alt="image" src="https://github.com/indohito/EE399/assets/107958888/d6bf23d8-7949-4d3f-b959-0fbb32391868">

## Summary 

Overall, the assignment explores the SHRED model, discusses important concepts in machine learning such as time lag and Gaussian noise, and conducts analyses to understand the impact of these factors on model performance.
