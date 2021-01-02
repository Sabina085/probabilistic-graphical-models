import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn import mixture
from EM_diagonal_covariance import EM_diagonal_covariance


np.random.seed(1000)
colors = ['ob', '*r']
labels = ['x', '+']

mean1 = [0, 0]
cov1 = [[10, 0], [0, 10]]
mean2 = [7, 7]
cov2 = [[1, 0], [0, 1]]

x1, y1 = np.random.multivariate_normal(mean1, cov1, 5000).T
x2, y2 = np.random.multivariate_normal(mean2, cov2, 5000).T

f_original_ds = plt.figure()
plt.plot(x1, y1, 'ob')
plt.plot(x2, y2, '*r')
plt.title('Original data (sampled from 2 Gaussians)')
plt.show()
plt.clf()
plt.close()
f_original_ds.savefig("Original_data.pdf", bbox_inches ='tight')

data = np.vstack([x1, y1]).T
data = np.vstack([data, np.vstack([x2, y2]).T])
random.shuffle(data)
print(data.shape)

kmeans = KMeans(n_clusters = 2)
KModel = kmeans.fit(data)
centroids = KModel.cluster_centers_
kmeans_predictions = KModel.predict(data)
print('centroids: ', centroids)


EM_full = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(data)
EM_predictions_full = EM_full.predict(data)
ps, means, covariances, responsabilities = EM_diagonal_covariance(data.T, 2)

EM_diag_predictions = np.zeros((data.shape[0], 1))

for n in range(data.shape[0]):
    max_resp = 0.0
    cluster_index = 0
    for k in range(2):
        if responsabilities[n, k] > max_resp:
            cluster_index = k
            max_resp = responsabilities[n, k]

    EM_diag_predictions[n] = cluster_index


fig_K_means_result = plt.figure()
idx_clusters = []
for j in range(2):
    idx_clusters.append([idx for idx in range(len(data)) if kmeans_predictions[idx] == j])
for j in range(2):
    plt.plot(data[idx_clusters[j]][:, 0], data[idx_clusters[j]][:, 1], colors[j])
plt.title('KMeans results')
plt.show()
plt.clf()
plt.close()
fig_K_means_result.savefig("KM_Synthetic_ds.pdf", bbox_inches ='tight')


fig_EM_diag_result = plt.figure()
idx_clusters = []
for j in range(2):
    idx_clusters.append([idx for idx in range(len(data)) if EM_diag_predictions[idx] == j])
for j in range(2):
    plt.plot(data[idx_clusters[j]][:, 0], data[idx_clusters[j]][:, 1], colors[j])
plt.title('EM for Gaussian Mixtures with diagonal covariance matrices')
plt.show()
plt.clf()
plt.close()
fig_EM_diag_result.savefig("EM_diag_Synthetic_ds.pdf", bbox_inches ='tight')


fig_EM_full_result = plt.figure()
idx_clusters = []
for j in range(2):
    idx_clusters.append([idx for idx in range(len(data)) if EM_predictions_full[idx] == j])
for j in range(2):
    plt.plot(data[idx_clusters[j]][:, 0], data[idx_clusters[j]][:, 1], colors[j])
plt.title('EM for Gaussian Mixtures with full covariance matrices')
plt.show()
plt.clf()
plt.close()
fig_EM_full_result.savefig("EM_full_Synthetic_ds.pdf", bbox_inches ='tight')
