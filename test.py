import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.metrics import confusion_matrix

import sys
import time

######################### Inputs ##################################

# The inputs can be passed as npy files here

images = np.load(sys.argv[1])           # data set X.       Expected dimensions(X,150,150)
labels = np.load(sys.argv[2]).T         # desired output y. Expected dimensions(y,)


########################## Load saved model ####################################
saved_model = "saved_model.pt"

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, 1)
        self.conv2 = nn.Conv2d(10, 20, 3, 1)
        self.conv3 = nn.Conv2d(20, 30, 3, 1)
        self.fc1 = nn.Linear(17*17*30, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 25)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.avg_pool2d(X, 2)
        X = F.relu(self.conv2(X))
        X = F.avg_pool2d(X, 2)
        X = F.relu(self.conv3(X))
        X = F.avg_pool2d(X, 2)
        X = X.view(-1, 17*17*30)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

    
torch.manual_seed(101)

CNNmodel = ConvolutionalNetwork()
CNNmodel.load_state_dict(torch.load(saved_model))
CNNmodel.eval()


########################## Test function definition ####################################

def normalize_data(X):
    mu = np.mean(X)
    std = np.std(X)
    return ((X-mu)/std)

# Tests the model for a given set of images, labels and prints the accuracy and confusion matrix
# returns a list of predicted labels for each X
def test(X_test, y_test):
    
    # Set parameters
    batch_size = 10
    
    X_test = X_test.reshape(len(X_test),1,150,150)
    
    # Normalization of data
    X_test = normalize_data(X_test)
    
    
    # Converting input numpy array to pytorch tensors
    X_testTensor = torch.Tensor(X_test)
    y_testTensor = torch.Tensor(y_test)
    
    y_testTensor = y_testTensor.type(torch.LongTensor)
    
    # Create test dataset and loader
    test_data = TensorDataset(X_testTensor,y_testTensor) 
    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True) 
    
    
    print(f'Test images available: {len(test_data)}')
    
    # Testing start time
    start_time = time.time()
    
    test_correct = 0
    final_predicted = []
    final_actual = []

    # For each test batch
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):

            # Apply the model
            y_pred = CNNmodel(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            test_correct += (predicted == y_test).sum()
            
            final_predicted += predicted
            final_actual += y_test
    
    print(f'Accuracy: {test_correct*(100/len(test_data))}')

    print(confusion_matrix(final_actual,final_predicted))
    
    print(f'\nDuration for testing: {time.time() - start_time:.0f} seconds') # print the time elapsed for testing
    
    return final_predicted
############################## Run test with data ############################

test(images, labels)

    

