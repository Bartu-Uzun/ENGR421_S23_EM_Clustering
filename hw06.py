import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[+0.0, +5.5],
                        [+0.0, +0.0],
                        [+0.0, -5.5]])

group_covariances = np.array([[[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+3.2, +2.8],
                               [+2.8, +3.2]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]]])

# read data into memory
data_set = np.genfromtxt("hw06_data_set.csv", delimiter = ",")



# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 3

# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    # your implementation starts below

    N = X.shape[0]

    #print("X: ", X.shape)
    means = np.genfromtxt("hw06_initial_centroids.csv", delimiter = ",")

    #print("means", means.shape)

    distance = dt.cdist(means, X)

    #print("distance: ", distance.shape)

    memberships = np.argmin(distance, axis=0)

    #print("memberships: ", memberships)

    priors = np.stack([np.mean(memberships == k) for k in range(K)])

    #print("priors: ", priors)

    #print(means[0])
    #print("---")
    #print(X[memberships == 0][:5])
    #print("-----")
    #print((X[memberships == 0] - means[0])[:5])

    #print(((X[memberships == 0] - means[0]) @ (X[memberships == 0] - means[0]).T).shape)

    covariances = np.zeros((K, 2, 2))
    for k in range(K):
        num = np.zeros((2,2))
        denum = 0
        
        for i in range(N):
            if memberships[i] == k:
                #print((X[i] - means[k])[:, None].shape)
                #print("1st: ", (X[i] - means[k])[:, None])
                #print("2nd: ", (X[i] - means[k])[:, None].T)
                #print("mult: ", (X[i] - means[k])[:, None] @ (X[i] - means[k])[:, None].T)
                num += (X[i] - means[k])[:, None] @ (X[i] - means[k])[:, None].T
                #print("num: \n", num)
                denum +=1

        #print("num: ", num)
        #print("i: ", i)
        
        covariances[k] = num / denum

    #print("covariances: ", covariances)



        
    # your implementation ends above
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below

    N = X.shape[0]
    
    for j in range(100):
        

        # fill the membership
        H = np.zeros((N, K))
        for i in range(N):

            denum = 0

            #print("covs, i: \n", covariances, i, " ", j)
            for k in range(K):
                #print("cov: \n", covariances[k])
                #print("det: \n", linalg.det(covariances[k]))
                denum += (1 / math.sqrt(2 * math.pi * linalg.det(covariances[k]))) * np.exp(- 0.5 * (X[i] - means[k])[:, None].T @ linalg.inv(covariances[k]) @ (X[i] - means[k])[:, None]) * priors[k]

            for k in range(K):
                num = (1 / math.sqrt(2 * math.pi * linalg.det(covariances[k]))) * np.exp(- 0.5 * (X[i] - means[k])[:, None].T @ linalg.inv(covariances[k]) @ (X[i] - means[k])[:, None]) * priors[k]
                H[i][k] = num / denum

        

        # update the parameters
        priors = np.stack(np.sum(H, axis=0)) / N

        means = np.zeros((3, 2))
        for k in range(K):
            means[k] = np.sum(H[i][k] * X[i] for i in range(N)) / np.sum(H[i][k] for i in range(N))


        #print("covs, j: \n", covariances, " ", j)
        covariances = np.zeros((K, 2, 2))
        for k in range(K):
            num = np.zeros((2,2))
            denum = 0
            for i in range(N):
                num += H[i][k] * (X[i] - means[k])[:, None] @ (X[i] - means[k])[:, None].T
                denum += H[i][k]
            covariances[k] = num / denum

        #print("covs, j: \n", covariances, " ", j)


    assignments = np.argmax(H, axis=1)
    # your implementation ends above
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below

    x1_interval = np.linspace(-8, +8, 801)
    x2_interval = np.linspace(-8, +8, 801)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
    X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T


    class_colors = ["#1f78b4", "#33a02c", "#e31a1c"]

    #plt.figure(figsize = (8, 8))
    for k in range(K):
        plt.plot(X[assignments == k, 0], X[assignments == k, 1], ".", markersize = 10, color = class_colors[k])
        D_og = stats.multivariate_normal.pdf(X_grid, mean = group_means[k, :], cov = group_covariances[k, :, :])
        D_og = D_og.reshape((len(x1_interval), len(x2_interval)))
        plt.contour(x1_grid, x2_grid, D_og, levels = [0.01], colors = "black", linestyles = "dashed")

        D_find = stats.multivariate_normal.pdf(X_grid, mean = means[k, :], cov = covariances[k, :, :])
        D_find = D_find.reshape((len(x1_interval), len(x2_interval)))
        plt.contour(x1_grid, x2_grid, D_find, levels = [0.01], colors = class_colors[k], linestyles = "solid")

    

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.show()
    # your implementation ends above
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)

