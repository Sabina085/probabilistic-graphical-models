import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import load_iris
from sklearn import mixture
from scipy import linalg


colors = ['ob', '*r', '^g', '+y']
    
iris = load_iris()

no_clusters = [2, 3, 4]
path = './Results/'


for i in no_clusters: # iterate over the number of clusters
    print("~~~~~~~~ K = " + str(i) + "~~~~~~~~~~~~~~")
    EM = mixture.GaussianMixture(n_components=i, covariance_type='full').fit(iris.data)
    predictions = EM.predict(iris.data)
    scores = EM.score_samples(iris.data)
    proba_for_every_cluster = EM.predict_proba(iris.data)
    
    points_cluster_x1_all = []
    points_cluster_x2_all = []
    points_cluster_x3_all = []
    points_cluster_x4_all = []
    
    for j in range(i): # iterate over the cluster indexes
        points_cluster_idxs = [idx for idx in range(len(iris.data)) if predictions[idx] == j] # take the indexes of the points from the cluster j
        points_clusters = np.array(iris.data[points_cluster_idxs]) # take the points from that cluster
        print("Cluster " + str(j) + ": ") 
        print(points_clusters.shape)
       
        points_cluster_x1 = points_clusters[:, 0] 
        points_cluster_x2 = points_clusters[:, 1]
        points_cluster_x3 = points_clusters[:, 2]
        points_cluster_x4 = points_clusters[:, 3]     
                
        points_cluster_x1_all.append(points_cluster_x1)
        points_cluster_x2_all.append(points_cluster_x2)
        points_cluster_x3_all.append(points_cluster_x3)
        points_cluster_x4_all.append(points_cluster_x4)
      

    f_x1_x2 = plt.figure()
    for j in range(i): # iterate over the cluster indexes
        plt.plot(points_cluster_x1_all[j], points_cluster_x2_all[j], colors[j], markersize=3) # plot the points from cluster j
    title_x1_x2 = "EM full covariance matrix - " + str(i) + " clusters (x1, x2)"
    f_x1_x2.suptitle(title_x1_x2)
    plt.xlabel('x1 (sepal length) ', fontsize = 10)
    plt.ylabel('x2 (sepal width) ', fontsize = 10)
    plt.show()
    plt.clf()
    plt.close()
    f_x1_x2.savefig(path + title_x1_x2 + ".pdf", bbox_inches ='tight')
    

    f_x1_x3 = plt.figure()
    for j in range(i): 
        plt.plot(points_cluster_x1_all[j], points_cluster_x3_all[j], colors[j], markersize = 3) 
    title_x1_x3 = "EM full covariance matrix - " + str(i) + " clusters (x1, x3)"
    f_x1_x3.suptitle(title_x1_x3)
    plt.xlabel('x1 (sepal length) ', fontsize = 10)
    plt.ylabel('x3 (petal length) ', fontsize = 10)
    plt.show()
    plt.clf()
    plt.close()
    f_x1_x3.savefig(path + title_x1_x3 + ".pdf", bbox_inches = 'tight')
    
    
    f_x1_x4 = plt.figure()
    for j in range(i): 
        plt.plot(points_cluster_x1_all[j], points_cluster_x4_all[j], colors[j], markersize = 3)
    title_x1_x4 = "EM full covariance matrix - " + str(i) + " clusters (x1, x4)"
    f_x1_x4.suptitle(title_x1_x4)
    plt.xlabel('x1 (sepal length) ', fontsize = 10)
    plt.ylabel('x4 (petal width) ', fontsize = 10)
    plt.show()
    plt.clf()
    plt.close()
    f_x1_x4.savefig(path + title_x1_x4 + ".pdf", bbox_inches = 'tight')
    

    f_x2_x3 = plt.figure()
    for j in range(i): 
        plt.plot(points_cluster_x2_all[j], points_cluster_x3_all[j], colors[j], markersize = 3) 
    title_x2_x3 = "EM full covariance matrix - " + str(i) + " clusters (x2, x3)"
    f_x2_x3.suptitle(title_x2_x3)
    plt.xlabel('x2 (sepal width) ', fontsize = 10)
    plt.ylabel('x3 (petal length) ', fontsize = 10)
    plt.show()
    plt.clf()
    plt.close()
    f_x2_x3.savefig(path + title_x2_x3 + ".pdf", bbox_inches = 'tight')
    
    
    f_x2_x4 = plt.figure()
    for j in range(i): 
        plt.plot(points_cluster_x2_all[j], points_cluster_x4_all[j], colors[j], markersize = 3)
    title_x2_x4 = "EM full covariance matrix - " + str(i) + " clusters (x2, x4)"
    f_x2_x4.suptitle(title_x2_x4)
    plt.xlabel('x2 (sepal width) ', fontsize = 10)
    plt.ylabel('x4 (petal width) ', fontsize = 10)
    plt.show()
    plt.clf()
    plt.close()
    f_x2_x4.savefig(path + title_x2_x4 + ".pdf", bbox_inches = 'tight')
    

    f_x3_x4 = plt.figure()
    for j in range(i): 
        plt.plot(points_cluster_x3_all[j], points_cluster_x4_all[j], colors[j], markersize = 3)  
    title_x3_x4 = "EM full covariance matrix - " + str(i) + " clusters (x3, x4)"
    f_x3_x4.suptitle(title_x3_x4)
    plt.xlabel('x3 (petal length) ', fontsize = 10)
    plt.ylabel('x4 (petal width) ', fontsize = 10)
    plt.show()
    plt.clf()
    plt.close()
    f_x3_x4.savefig(path + title_x3_x4 + ".pdf", bbox_inches = 'tight')   
