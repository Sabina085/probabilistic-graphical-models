import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import load_iris
from sklearn import mixture
import scipy
from scipy import linalg


colors = ['ob', '*r', '^g', '+y']
path = './Results/'

    
def compute_likelihood(X, no_components, ps, covariances, means):
    number_examples = X.shape[1]
    log_likelihood = 0.0
    for i in range(number_examples):
        sum_k_comp = 0.0
        for k in range(no_components):
            p_x_given_z = scipy.stats.multivariate_normal.pdf(X[:, i], means[:, k], covariances[k])
            sum_k_comp += (ps[k] * p_x_given_z)
            
        log_likelihood += np.log(sum_k_comp)
        
    return log_likelihood

    
def EM_diagonal_covariance(X, no_components):
    dim = X.shape[0]
    no_examples = X.shape[1]

    # ~~~ Step 1 -> initialization ~~~
    ps = np.zeros((no_components, 1))
    ps.fill(1./no_components)

    means = np.random.rand(dim, no_components)
    covariances = []
    for k in range(no_components):
        covariance = np.eye(dim) 
        covariances.append(covariance)
 
    log_likelihood = compute_likelihood(X, no_components, ps, covariances, means)
    print('log_likelihood: ', log_likelihood)
    
    eps = 1e-4
    dif = 10
    
    responsabilities = np.zeros((no_examples, no_components))

    while dif > eps:
        # ~~~~Expectation Step~~~~
        for n in range(no_examples):
            total_prob = 0.0
            for k in range(no_components):           
                responsabilities[n, k] = ps[k] * scipy.stats.multivariate_normal.pdf(X[:, n], means[:, k], covariances[k])
                total_prob += responsabilities[n, k]
            responsabilities[n, :] /= total_prob

            assert responsabilities[n, :].sum() - 1 < 1e-4        

        # ~~~~Maximization Step~~~~
        Nks = np.zeros((no_components, 1))
        for k in range(no_components):
            for n in range(no_examples):
                Nks[k] += responsabilities[n, k]

        
        means = np.zeros((dim, no_components))

        # Compute the means and the ps        
        for k in range(no_components):
            for n in range(no_examples):
                means[:, k] += (responsabilities[n, k] * X[:, n])
            means[:, k] /= Nks[k]
            ps[k] = Nks[k] / no_examples

        
        # Compute the covariances
        sigmas = np.zeros((dim, no_components))
        for k in range(no_components):
            for j in range(dim):
                for n in range(no_examples):
                    sigmas[j, k] += (responsabilities[n, k] * (X[j, n] - means[j, k]) * (X[j, n] - means[j, k]))            
                sigmas[j, k] /= Nks[k]

            covariances[k] = np.diag(sigmas[:, k])
                        
        log_likelihood_new = compute_likelihood(X, no_components, ps, covariances, means)
        dif = np.linalg.norm(log_likelihood_new - log_likelihood)
        log_likelihood = log_likelihood_new
        print('log_likelihood: ', log_likelihood)
    
    return ps, means, covariances, responsabilities
                    

iris = load_iris()
print(np.asarray(iris.data).shape)
X = np.asarray(iris.data).T
print(X.shape)

no_clusters = [2, 3, 4]

for i in no_clusters:
    print('~~~~~~~ K = ', str(i), ' ~~~~~~~~~')
    ps, means, covariances, responsabilities = EM_diagonal_covariance(X, i)

    predictions = np.zeros((X.shape[1], 1))

    for n in range(X.shape[1]):
        max_resp = 0.0
        cluster_index = 0
        for k in range(i):
            if responsabilities[n, k] > max_resp:
                cluster_index = k
                max_resp = responsabilities[n, k]

        predictions[n] = cluster_index

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
        plt.plot(points_cluster_x1_all[j], points_cluster_x2_all[j], colors[j], markersize=3) # plot the points from the cluster j
    title_x1_x2 = "EM diagonal covariance matrix - " + str(i) + " clusters (x1, x2)"
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
    title_x1_x3 = "EM diagonal covariance matrix - " + str(i) + " clusters (x1, x3)"
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
    title_x1_x4 = "EM diagonal covariance matrix - " + str(i) + " clusters (x1, x4)"
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
    title_x2_x3 = "EM diagonal covariance matrix - " + str(i) + " clusters (x2, x3)"
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
    title_x2_x4 = "EM diagonal covariance matrix - " + str(i) + " clusters (x2, x4)"
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
    title_x3_x4 = "EM diagonal covariance matrix - " + str(i) + " clusters (x3, x4)"
    f_x3_x4.suptitle(title_x3_x4)
    plt.xlabel('x3 (petal length) ', fontsize = 10)
    plt.ylabel('x4 (petal width) ', fontsize = 10)
    plt.show()
    plt.clf()
    plt.close()
    f_x3_x4.savefig(path + title_x3_x4 + ".pdf", bbox_inches = 'tight')
    