import numpy as np
import matplotlib.pyplot as plt
import glob
import re


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
       
     
    x_train_data = x_train_data.T 
    y_train_data = y_train_data.T 
    
    print('~~~Train ',  fileNames[i], '~~~')
    
    N_train = len(y_train_data[0])
    pi = np.sum(y_train_data) / N_train
    n_0 = np.sum(1.0-y_train_data)
    
    miu_0 = np.sum((1.0-y_train_data)*x_train_data, axis = 1, keepdims=True)/n_0 
    print('miu_0 = ', miu_0)
    
    n_1 = np.sum(y_train_data)
    miu_1 = np.sum(y_train_data*x_train_data, axis = 1, keepdims=True)/n_1 
    print('miu_1 = ', miu_1)
        
    t0 = (x_train_data - miu_0) * (1-y_train_data)
    cov_0 = (np.matmul(t0, t0.T)) / n_0
    print('cov_0 = ', cov_0)
    
    t1 = (x_train_data - miu_1) * y_train_data
    cov_1 = (np.matmul(t1, t1.T)) / n_1
    print('cov_1 = ', cov_1)
    
    det_cov_0 = cov_0[0, 0]*cov_0[1, 1] - cov_0[0, 1]*cov_0[1, 0]
    det_cov_1 = cov_1[0, 0]*cov_1[1, 1] - cov_1[0, 1]*cov_1[1, 0]
    
    free_term = np.log(pi / (1. - pi)) - (1./2.) * np.log(det_cov_1) + (1./2.) * np.log(det_cov_0)
    prec_matrix_0 = np.linalg.inv(cov_0)
    prec_matrix_1 = np.linalg.inv(cov_1)

    max1_train = int(max(x_train_data_pos_1 + x_train_data_neg_1)) + 2
    max2_train = int(max(x_train_data_pos_2 + x_train_data_neg_2)) + 2 
    min1_train = int(min(x_train_data_pos_1 + x_train_data_neg_1)) - 2
    min2_train = int(min(x_train_data_pos_2 + x_train_data_neg_2)) - 2
    

    x1 = np.linspace(min1_train, max1_train, 1000)
    x2 = np.linspace(min2_train, max2_train, 1000)
    
    x1, x2 = np.meshgrid(x1,x2)

    term_0 = (1./2.) * (prec_matrix_0[0, 0]*((x1 - miu_0[0, 0])**2) + prec_matrix_0[1, 0]*(x1 - miu_0[0, 0])*(x2 - miu_0[1, 0]) + prec_matrix_0[0, 1]*(x1 - miu_0[0, 0])*(x2 - miu_0[1, 0]) + prec_matrix_0[1, 1]*((x2 - miu_0[1, 0])**2))
    term_1 = (1./2.) * (prec_matrix_1[0, 0]*((x1 - miu_1[0, 0])**2) + prec_matrix_1[1, 0]*(x1 - miu_1[0, 0])*(x2 - miu_1[1, 0]) + prec_matrix_1[0, 1]*(x1 - miu_1[0, 0])*(x2 - miu_1[1, 0]) + prec_matrix_1[1, 1]*((x2 - miu_1[1, 0])**2))
    

    plt.axis([min1_train, max1_train, min2_train, max2_train])
    plt.title("QDA - " + "train " + fileNames[i])  
    plt.plot(x_train_data_pos_1, x_train_data_pos_2, 'go', x_train_data_neg_1, x_train_data_neg_2, 'bo')
    plt.contour(x1, x2,(term_0 - term_1 + free_term), [0], colors='r')
    plt.savefig('QDA_train' +  fileNames[i] +'.pdf')
    plt.close()
 
    nr_correct_train = N_train
        
    for j in range(x_train_data.shape[1]):
        x1 = x_train_data[0, j]
        x2 = x_train_data[1, j]
        term_0 = (1./2.) * (prec_matrix_0[0, 0]*((x1 - miu_0[0, 0])**2) + prec_matrix_0[1, 0]*(x1 - miu_0[0, 0])*(x2 - miu_0[1, 0]) + prec_matrix_0[0, 1]*(x1 - miu_0[0, 0])*(x2 - miu_0[1, 0]) + prec_matrix_0[1, 1]*((x2 - miu_0[1, 0])**2))
        term_1 = (1./2.) * (prec_matrix_1[0, 0]*((x1 - miu_1[0, 0])**2) + prec_matrix_1[1, 0]*(x1 - miu_1[0, 0])*(x2 - miu_1[1, 0]) + prec_matrix_1[0, 1]*(x1 - miu_1[0, 0])*(x2 - miu_1[1, 0]) + prec_matrix_1[1, 1]*((x2 - miu_1[1, 0])**2))
        y = sigmoid(term_0 - term_1 + free_term)
        if ((y > 0.5) and (y_train_data[0, j] != 1.0)) or ((y <= 0.5) and (y_train_data[0, j] != 0.0)):
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
            
            
    x_test_data = x_test_data.T 
    y_test_data = y_test_data.T
        
        
    max1_test = int(max(x_test_data_pos_1 + x_test_data_neg_1)) + 2
    max2_test = int(max(x_test_data_pos_2 + x_test_data_neg_2)) + 2
    min1_test = int(min(x_test_data_pos_1 + x_test_data_neg_1)) - 2
    min2_test = int(min(x_test_data_pos_2 + x_test_data_neg_2)) - 2
    
        
    x1 = np.linspace(min1_test, max1_test, 1000)
    x2 = np.linspace(min2_test, max2_test, 1000)
    
    x1, x2 = np.meshgrid(x1,x2)

        
    term_0 = (1./2.) * (prec_matrix_0[0, 0]*((x1 - miu_0[0, 0])**2) + prec_matrix_0[1, 0]*(x1 - miu_0[0, 0])*(x2 - miu_0[1, 0]) + prec_matrix_0[0, 1]*(x1 - miu_0[0, 0])*(x2 - miu_0[1, 0]) + prec_matrix_0[1, 1]*((x2 - miu_0[1, 0])**2))
    term_1 = (1./2.) * (prec_matrix_1[0, 0]*((x1 - miu_1[0, 0])**2) + prec_matrix_1[1, 0]*(x1 - miu_1[0, 0])*(x2 - miu_1[1, 0]) + prec_matrix_1[0, 1]*(x1 - miu_1[0, 0])*(x2 - miu_1[1, 0]) + prec_matrix_1[1, 1]*((x2 - miu_1[1, 0])**2))    

    plt.axis([min1_test, max1_test, min2_test, max2_test])
    plt.title("QDA - " + "test " + fileNames[i])
    plt.plot(x_test_data_pos_1, x_test_data_pos_2, 'go', x_test_data_neg_1, x_test_data_neg_2, 'bo')
    plt.contour(x1, x2,(term_0 - term_1 + free_term), [0], colors='r')
    plt.savefig('QDA_test' +  fileNames[i] +'.pdf')
    plt.close()
    
    
    N_test = x_test_data.shape[1]
    nr_correct_test = N_test
        
    for j in range(x_test_data.shape[1]):
        x1 = x_test_data[0, j]
        x2 = x_test_data[1, j]
        term_0 = (1./2.) * (prec_matrix_0[0, 0]*((x1 - miu_0[0, 0])**2) + prec_matrix_0[1, 0]*(x1 - miu_0[0, 0])*(x2 - miu_0[1, 0]) + prec_matrix_0[0, 1]*(x1 - miu_0[0, 0])*(x2 - miu_0[1, 0]) + prec_matrix_0[1, 1]*((x2 - miu_0[1, 0])**2))
        term_1 = (1./2.) * (prec_matrix_1[0, 0]*((x1 - miu_1[0, 0])**2) + prec_matrix_1[1, 0]*(x1 - miu_1[0, 0])*(x2 - miu_1[1, 0]) + prec_matrix_1[0, 1]*(x1 - miu_1[0, 0])*(x2 - miu_1[1, 0]) + prec_matrix_1[1, 1]*((x2 - miu_1[1, 0])**2))
        y = sigmoid(term_0 - term_1 + free_term)

        if ((y > 0.5) and (y_test_data[0, j] != 1.0)) or ((y <= 0.5) and (y_test_data[0, j] != 0.0)):
            nr_correct_test -=1.0
    

    accuracy_test = nr_correct_test/N_test
    
    print('~~~Test ',  fileNames[i], '~~~')
    print('Accuracy test' + fileNames[i] + ' = ' + str(accuracy_test*100.0) + ('%'))
    print('Misclassification error test' + fileNames[i] + ' = ' + str(round((1-accuracy_test)*100, 2)) + ('%'))    
                
    