import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import math


dataPath = "data/";

fileNames = ["A", "B", "C"]


def sigmoid(x):
    return 1./(1+np.exp(-x))


for i in range(len(fileNames)):
    trainFilePath = dataPath + "train" + fileNames[i]
    testFilePath = dataPath + "test" + fileNames[i]
    trainDataFile = glob.glob(trainFilePath)[0]
    trainFile = open(trainDataFile, "r")
    xs = []
    ys = []
    for line in trainFile:
        data = line.split()
        xs.append(data[0])
        xs.append(data[1])
        ys.append(data[2])
    
    x_train_data = np.zeros((int(len(xs)/2), 2))
    y_train_data = np.zeros((int(len(ys)), 1))
    x_train_data_pos_1 = []
    x_train_data_pos_2 = []
    x_train_data_neg_1 = []
    x_train_data_neg_2 = []
    
    for j in range(0, x_train_data.shape[0]):
        x_train_data[j, 0] = xs[2*j]
        x_train_data[j, 1] = xs[2*j+1]
        ys[j] = float(re.sub('["]', '', ys[j]))
        y_train_data[j, 0] = ys[j]
        if ys[j] == 1.0:
            x_train_data_pos_1.append(float(xs[2*j]))
            x_train_data_pos_2.append(float(xs[2*j+1]))
        else:
            x_train_data_neg_1.append(float(xs[2*j]))
            x_train_data_neg_2.append(float(xs[2*j+1]))
            
                    
    x_train_data_bias = np.hstack((np.ones((x_train_data.shape[0], 1)), x_train_data))
    W_and_bias = np.expand_dims(np.random.uniform(-0.01, 0.01, x_train_data_bias.shape[1]), axis=1)
    
    eps = 1e-3
    norm = math.inf
    minibatch_size = 100 # 16
    lr = 0.1 # 0.03
    iter = 0
    
    while(norm > eps  and iter < 15000):
        
        iter += 1
        indexes_for_minibatch = range(0, 100) #np.random.randint(0, high= x_train_data_bias.shape[0], size=minibatch_size) #range(0, 100) 
        X = x_train_data_bias[indexes_for_minibatch, :]
        Y_pred = sigmoid (np.dot(X, W_and_bias))
        Y_gt = np.expand_dims(y_train_data[indexes_for_minibatch, 0], axis=1)
        loss = np.squeeze((-1./minibatch_size)*(np.dot(Y_gt.T, np.log2(Y_pred) + np.dot((1-Y_gt).T, np.log2(1-Y_pred)))))

        grad = (1./minibatch_size) * np.dot(X.T, (sigmoid(np.dot(X, W_and_bias)) - Y_gt))
        W_and_bias_new = W_and_bias - lr*grad
        
        norm = np.linalg.norm(grad) 
                
        W_and_bias = W_and_bias_new
        
        
    print('~~~Train ',  fileNames[i], '~~~')
    
    W = W_and_bias[1:3, :]
    b = W_and_bias[0, :]
    
    print('W = ', W) #(2, 1)
    print('b = ', b)
          
    
    max1_train = int(max(x_train_data_pos_1 + x_train_data_neg_1)) + 2
    max2_train = int(max(x_train_data_pos_2 + x_train_data_neg_2)) + 2
    min1_train = int(min(x_train_data_pos_1 + x_train_data_neg_1)) - 2
    min2_train = int(min(x_train_data_pos_2 + x_train_data_neg_2)) - 2
    
    x1 = np.linspace(min1_train, max1_train, 1000)
    x2 = np.squeeze((0.5-b[0] - W[0, 0]*x1)/W[1, 0])
    
    plt.axis([min1_train, max1_train, min2_train, max2_train])
    plt.title("Logistic regression - " + "train " + fileNames[i])  
    plt.plot(x_train_data_pos_1, x_train_data_pos_2, 'go', x_train_data_neg_1, x_train_data_neg_2, 'bo', x1, x2, 'r-', linewidth=1)
    plt.savefig('Logistic_regression_train' +  fileNames[i] +'.pdf')
    plt.close()

    N_train = y_train_data.shape[0]
    nr_correct_train = N_train
        
    for j in range(x_train_data.shape[0]):
        y =  np.squeeze(sigmoid(W[0, 0] * x_train_data[j, 0] + W[1, 0]*x_train_data[j, 1] + b[0]))
        if ((y > 0.5) and (y_train_data[j, 0] != 1.0)) or ((y <= 0.5) and (y_train_data[j, 0] != 0.0)):
            nr_correct_train -=1.0
    
    accuracy_train = nr_correct_train/N_train
    
    print('Accuracy train' + fileNames[i] + ' = ' + str(accuracy_train*100.0) + ('%'))
    print('Misclassification error train' + fileNames[i] + ' = ' + str(round((1-accuracy_train)*100, 2)) + ('%'))    


    testDataFile = glob.glob(testFilePath)[0]
    testFile = open(testDataFile, "r")
    xs = []
    ys = []
    for line in testFile:
        data = line.split()
        xs.append(data[0])
        xs.append(data[1])
        ys.append(data[2])
        
    x_test_data = np.zeros((int(len(xs)/2), 2))
    y_test_data = np.zeros((int(len(ys)), 1))     
    x_test_data_pos_1 = []
    x_test_data_pos_2 = []
    x_test_data_neg_1 = []
    x_test_data_neg_2 = []
    
    for j in range(0, int(len(xs)/2)):
        x_test_data[j, 0] = xs[2*j]
        x_test_data[j, 1] = xs[2*j+1]
        ys[j] = float(re.sub('["]', '', ys[j]))
        y_test_data[j, 0] = ys[j]
        if ys[j] == 1.0:
            x_test_data_pos_1.append(float(xs[2*j]))
            x_test_data_pos_2.append(float(xs[2*j+1]))
        else:
            x_test_data_neg_1.append(float(xs[2*j]))
            x_test_data_neg_2.append(float(xs[2*j+1]))            
        

    max1_test = int(max(x_test_data_pos_1 + x_test_data_neg_1)) + 2
    max2_test = int(max(x_test_data_pos_2 + x_test_data_neg_2)) + 2
    min1_test = int(min(x_test_data_pos_1 + x_test_data_neg_1)) - 2
    min2_test = int(min(x_test_data_pos_2 + x_test_data_neg_2)) - 2
    

    plt.axis([min1_test, max1_test, min2_test, max2_test])
    plt.title("Logistic regression - " + "test " + fileNames[i])
    plt.plot(x_test_data_pos_1, x_test_data_pos_2, 'go', x_test_data_neg_1, x_test_data_neg_2, 'bo', x1, x2, 'r-', linewidth=1)
    plt.savefig('Logistic regression_test' +  fileNames[i] +'.pdf')
    plt.close()
    

    N_test = x_test_data.shape[0]
    nr_correct_test = N_test
        
    for j in range(x_test_data.shape[0]):
        y = np.squeeze(sigmoid(W[0, 0] * x_test_data[j, 0] + W[1, 0]*x_test_data[j, 1] + b[0]))
        if ((y > 0.5) and (y_test_data[j, 0] != 1.0)) or ((y <= 0.5) and (y_test_data[j, 0] != 0.0)):
            nr_correct_test -=1.0
    

    accuracy_test = nr_correct_test/N_test
    
    print('~~~Test ',  fileNames[i], '~~~')
    print('Accuracy test' + fileNames[i] + ' = ' + str(accuracy_test*100.0) + ('%'))
    print('Misclassification error test' + fileNames[i] + ' = ' + str(round((1-accuracy_test)*100, 2)) + ('%'))  
  