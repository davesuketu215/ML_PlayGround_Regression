from collections import defaultdict
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

INPUT_CONFIG_PATH = 'input_config.json'

with open(INPUT_CONFIG_PATH, 'r') as fp:
    config = json.load(fp)

lossValuesDict = defaultdict()      # Data Structure to hold loss values at each weight update

weightUpdateCount = 0               # Count how many times weight is updated until yet

# Initialize a Plot
plt.figure(figsize=(12,12))


def load_dataset():
    # Read config for path
    datasetPath = config['DatasetRelPath']
    
    # Read Data
    data = pd.read_csv(datasetPath)
    return data


def split_data_x_y(data):
    numCols = data.shape[1]
    x = data.iloc[:, :numCols-1]
    y = data.iloc[:, numCols-1:]
    
    return x, y


def find_loss_value(y_pred, y_actual):
    if config['LossFunction'] == 'MSE':
        loss = calculate_loss_mse(y_pred, y_actual)
    elif config['LossFunction'] == 'MAD':
        loss = calculate_loss_mad(y_pred, y_actual)
    elif config['LossFunction'] == 'LogLoss':
        loss = calculate_loss_logloss(y_pred, y_actual)
                
    return loss


def calculate_loss_mse(y_pred, y_actual):

    diff_series = y_pred.iloc[:,0].subtract(y_actual.iloc[:,0])
    mult_series = diff_series.multiply(diff_series)
    lossMSE     = mult_series.mul(0.5/len(y_actual))

    return lossMSE


def calculate_loss_mad(y_pred, y_actual):

    diff_series = y_pred.iloc[:,0].subtract(y_actual.iloc[:,0])
    div_series  = diff_series.mul(1/len(y_actual))
    lossMAD     = div_series.abs()

    return lossMAD


def calculate_loss_logloss(y_pred, y_actual):

    ones_array  = np.ones(len(y_actual))
    ones_series = pd.Series(ones_array)

    lossLogLoss = y_actual.iloc[:,0].multiply(np.log(y_pred.iloc[:,0])).add(ones_series.subtract(y_actual.iloc[:,0]).multiply(np.log(ones_series.subtract(y_pred.iloc[:,0]))))
    lossLogLoss = lossLogLoss.mul(-1/len(y_actual))
    # print('\nLoss Log Loss = \n', lossLogLoss)
    return lossLogLoss


def plot_regression_line(plot_regr_line, x_train, y_train, y_pred, m, c):
    if config['Algorithm'] == 'LinearRegression':
        plot_regr_line.set_title('Best-Fit Regression Line\nSlope(m): ' + str(round(m,2)) + '     Y-Int(c): ' + str(round(c,2)))
    elif config['Algorithm'] == 'LogisticRegression':
        plot_regr_line.set_title('Best-Fit Regression Line\nw1: ' + str(round(m,2)) + '     w0: ' + str(round(c,2)))
    
    plot_regr_line.scatter(x_train, y_train, color='blue')
    plot_regr_line.set_xlim(left=0)
    
    if config['Algorithm'] == 'LinearRegression':
        plot_regr_line.set_ylim(bottom=0, top=max(y_train)+1)
    elif config['Algorithm'] == 'LogisticRegression':
        plot_regr_line.set_ylim(bottom=0, top=1)
        plot_regr_line.axhline(y=config['ClassificationMinThreshold'])
    
    plot_regr_line.set_xlabel('X', loc='center')
    plot_regr_line.set_ylabel('Y', loc='center')
    
    if config['Algorithm'] == 'LinearRegression':
        plot_regr_line.plot([min(x_train), max(x_train)], [min(y_pred), max(y_pred)], color='red')
    elif config['Algorithm'] == 'LogisticRegression':  
        plot_regr_line.scatter(x_train, y_pred, color='red')  



def plot_loss_values(plot_loss_vals, n):
    global lossValuesDict
    
    lossDictKeysList = lossValuesDict.keys()
    lossDictValsList = list(lossValuesDict.values())
    
    plot_loss_vals.set_title('Loss Curve\nError: ' + str(round(lossDictValsList[-1],2)))
    plot_loss_vals.set_xlim(left=1, right=n*config['MaxEpochs'])
    
    if config['Algorithm'] == 'LinearRegression':
        plot_loss_vals.set_ylim(bottom=0, top=max(lossDictValsList))
    elif config['Algorithm'] == 'LogisticRegression':
        plot_loss_vals.set_ylim(bottom=0, top=max(lossDictValsList))
    
    plot_loss_vals.set_xticks(np.arange(0, n*config['MaxEpochs'], 100))
    plot_loss_vals.set_xlabel('Weight Update Iterations', loc='center')
    plot_loss_vals.set_ylabel('Loss Value', loc='center')
    plot_loss_vals.plot(lossDictKeysList, lossDictValsList, marker='o', markersize=2, color='red')



def plot_curves(x_train, y_train, y_pred, n, m, c):
    # Convert into python list
    x_train_list    = [elem[0] for elem in x_train.values.tolist()]
    y_train_list    = [elem[0] for elem in y_train.values.tolist()]
    y_pred_list     = [elem[0] for elem in y_pred.values.tolist()]

    # Placing the plots in the plane
    plot_regr_line  = plt.subplot2grid((1, 2), (0, 0), rowspan=1 ,colspan=1)    # Regression Line
    plot_loss_vals  = plt.subplot2grid((1, 2), (0, 1), rowspan=1 ,colspan=1)    # Error Function

    plot_regression_line(plot_regr_line, x_train_list, y_train_list, y_pred_list, m, c)
    plot_loss_values(plot_loss_vals, n)

    plt.show(block=False)
    plt.pause(5)
    plt.clf()
    

def perform_weight_update(x_train, y_train, y_pred, num_iter, w1, w0, k):
    global weightUpdateCount
    
    # Before performing any weight updates
    # Plot the curves for the present outcomes
    plot_curves(x_train, y_train, y_pred, num_iter, w1, w0)

    L = config['LearningRate']
    
    x_train = x_train.iloc[:,0]
    y_train = y_train.iloc[:,0]
    y_pred  = y_pred.iloc[:,0]
    n = len(x_train)                # Batch Size
    
    # For Stochastic Gradient Descent
    if k != None:
        x_train = x_train[k:k+1]
        y_train = y_train[k:k+1]
        y_pred  = y_pred[k:k+1]


    if config['LossFunction'] == 'MSE':
        D_w1 = (-1/n) * sum((y_train - y_pred) * x_train)  # Derivative wrt m
        D_w0 = (-1/n) * sum(y_train - y_pred)  # Derivative wrt c
        w1 = w1 - L * D_w1  # Update m
        w0 = w0 - L * D_w0  # Update c

    # elif config['LossFunction'] == 'MAD':
        
    elif config['LossFunction'] == 'LogLoss':
        D_w1 = (-1/n) * sum((y_train - y_pred) * x_train)
        D_w0 = (-1/n) * sum(y_train - y_pred)
        w1 = w1 - L * D_w1
        w0 = w0 - L * D_w0
             
    weightUpdateCount += 1

    return w1, w0


def predict_and_find_loss(w1, w0, x_train, y_train):

    if config['Algorithm'] == 'LinearRegression':
        y_pred = w1 * x_train + w0
    elif config['Algorithm'] == 'LogisticRegression':
        y_pred = 1/(1+np.exp((-1)*(w1 * x_train + w0)))
    else:
        print('Invalid Model Selection')

    # Find the loss (error) value
    loss = find_loss_value(y_pred, y_train)

    return y_pred, loss


def train_regression(x_train, y_train):
    global lossValuesDict

    # Initialize the model weights
    if config['Algorithm'] == 'LinearRegression':
        w1 = 0
        w0 = 0
    elif config['Algorithm'] == 'LogisticRegression':
        w1 = 0.5
        w0 = -5
    
    maxEpochs = config['MaxEpochs']

    for i in range(maxEpochs):
        # Initialize the loss value
        epochLoss = 0
        
        y_pred, loss = predict_and_find_loss(w1, w0, x_train, y_train)

        if config['Optimizer'] == 'StochasticGradientDescent':
            for k in range(len(x_train)):
                lossValuesDict[weightUpdateCount] = loss[k:k+1].to_numpy()[0]
                w1, w0 = perform_weight_update(x_train, y_train, y_pred, len(x_train), w1, w0, k)
                y_pred, loss = predict_and_find_loss(w1, w0, x_train, y_train)
                continue

        if config['Optimizer'] == 'GradientDescent':
            epochLoss = loss.to_numpy().sum()
            lossValuesDict[weightUpdateCount] = epochLoss
            w1, w0 = perform_weight_update(x_train, y_train, y_pred, 1, w1, w0, None)

    trainedModelWeightsDict = {
        "Slope":        w1,
        "YIntercept":   w0
    }

    return trainedModelWeightsDict
    

if __name__ == '__main__':
    # Load the dataset
    data_train = load_dataset()

    # Split data into Features and Target Values
    x_train, y_train = split_data_x_y(data_train)   # x_train, y_train are DataFrames

    trainedModelWeightsDict = train_regression(x_train, y_train)
    plt.show()
    print('Done')