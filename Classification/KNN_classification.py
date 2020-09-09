#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load the libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.io import loadmat
from sklearn import metrics
from pylab import *
import sklearn.preprocessing


# In[20]:


# Load dev data
d = loadmat('data_ivector_embeddings.mat') # load the data
data2 = d['devIVs']

# define X 
inputs = data2.transpose()

# define Y
labels = d['labels']
labels = labels[:,0]
labels

# target names = categories
target_names = ['audiobooks', 'broadcast_interview', 'child', 'clinical', 'court', 'maptask', 'meeting', 'restaurant', 'socio_field', 'socio_lab', 'webvideo']

# add names of files
list_file_names = ['DH_000'+str(i) for i in range(1,10)]
list_file_names += ['DH_00'+str(i) for i in range(10,100)]
list_file_names += ['DH_0'+str(i) for i in range(100,193)]

d['files'] = list_file_names


# In[21]:


#load eval data
labels_eval = pd.read_csv("domain_list_eval.csv") 
labels_eval.shape

eval_data = []

with open('dev_eval_dihard_ivector.txt', 'r') as f:
    for i in f.readlines():
        txt = i.split()
        if txt[0][0] == 'e':
            eval_data.append(txt)

eval_data.sort()
name_files_eval = [i[0][5:] for i in eval_data]
eval_data = [i[1:] for i in eval_data]


df_eval = pd.DataFrame(list(zip(eval_data,labels_eval['labels'], name_files_eval)),columns = ["data","label","files"])
df_eval = df_eval.sample(frac=1)


# In[27]:


# create dataframe which contains the data with the labels of dev data
df = pd.DataFrame(list(zip(inputs,labels, d['files'])),columns = ["data","label", 'files'])
df = df.sample(frac=1) # shuffle the data
df
# print head of dataframe
#df.shape


# In[28]:


# Define X and Y variables
X = list(df['data'])
Y = df['label']
Z = df['files']


# In[29]:


# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)

# len(X_train)  # 144
# len(X_test)  # 48
# len(y_train)  # 144
# len(y_test)  # 4
# TOTAL OF 192 files


# In[30]:


import operator

error_rate = []
scores = {}
scores_list = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    scores[i] = metrics.accuracy_score(y_test, pred_i)
    error_rate.append(np.mean(pred_i != y_test))
    
# Configure and plot error rate over k values
plt.figure(figsize=(10,4))
plt.plot(range(1,40), error_rate, color='blue', linestyle='solid', marker='.', markerfacecolor='black', markersize=10)
plt.title('Error Rate vs. K-Values')
plt.xlabel('K-Values')
plt.ylabel('Error Rate')

print(scores)
max_n = max(scores.items(), key=operator.itemgetter(1))[0]
print('\nMax neighbor = '+str(max_n)+' with accuracy of : ' , max(scores.values()))
# We can see that the best model is with 3 n_neighbors as parameters


# In[31]:


# Create the model with optimal value K = 4
knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')


# In[32]:


knn.fit(X, Y)


# In[33]:


# PREDICTION WITH THE TRAINING DATA
y_pred_train = knn.predict(X)

# display the y_pred and y_test label
for v in zip(Y, y_pred_train, Z):
    print (*v)

# create dataframe with information 
df_2 = pd.DataFrame()
df_2['dev_labels'] = Y
df_2['dev_labels_pred'] = y_pred_train
df_2['names'] = Z
df_2.sort_values('dev_labels_pred')
df_2.to_csv('dev_knn_filesivectors.csv')


# In[34]:


# display accuracy level of the prediction of train data
accuracy = metrics.accuracy_score(Y, y_pred_train)
accuracy


# In[35]:


# Print out classification report
print(classification_report(Y, y_pred_train))


# In[15]:


# Print out confusion matrix
cmat = confusion_matrix(Y, y_pred_train)
print(cmat)

# display True negative
print('TP - True Negative {}'.format(cmat[0,0]))
print('FP - False Positive {}'.format(cmat[0,1]))
print('FN - False Negative {}'.format(cmat[1,0]))
print('TP - True Positive {}'.format(cmat[1,1]))


# In[16]:


# EVALUATION DATA PREDICTION

X_eval = list(df_eval['data'])
Y_eval = df_eval['label']
Z_eval = df_eval['files']
y_pred_eval = knn.predict(X_eval)

# display the y_pred and y_test label
for v in zip(Y_eval, y_pred_eval, Z_eval):
    print (*v)

# create dataframe with information 
df_3 = pd.DataFrame()
df_3['eval_labels'] = Y_eval
df_3['eval_labels_pred'] = y_pred_eval
df_3['names'] = Z_eval
df_3.sort_values('eval_labels_pred')
df_3.to_csv('eval_knn_filesivectors.csv')


# In[17]:


# display accuracy level of the prediction of train data
accuracy = metrics.accuracy_score(Y_eval, y_pred_eval)
accuracy


# In[18]:


# Print out classification report
print(classification_report(Y_eval, y_pred_eval))


# In[19]:


# Print out confusion matrix
cmat = confusion_matrix(Y_eval, y_pred_eval)
print(cmat)

# display True negative
print('TP - True Negative {}'.format(cmat[0,0]))
print('FP - False Positive {}'.format(cmat[0,1]))
print('FN - False Negative {}'.format(cmat[1,0]))
print('TP - True Positive {}'.format(cmat[1,1]))


# In[27]:


#tsne visualization
def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P

def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) +                 (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            # print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y


print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")

# data = loadmat('data_xvector_embeddings.mat')
colors=["#FF2D00", "#FF9B00", "#FFEC00", "#80FF00", "#00FFD8", "#0087FF", "#E000FF", "#FF0078", "#3C435A", "#366F29", "#B58724"]

# inputs = X # change there to see train data
labels_pred=y_pred

X_2=sklearn.preprocessing.normalize(eval_data, norm='l2', axis=0)
Y_2 = tsne(X_2, 2, 20, 20.0)

f, ax = plt.subplots(1)
for i in np.unique(labels_pred):
    mask = labels_pred == i
    plt.scatter(Y_2[mask, 0], Y_2[mask, 1], 20, label=target_names[i-1], c=colors[i-1])

for label, x, y in zip(labels, Y_2[:, 0], Y_2[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-0.5, 0.5),
        textcoords='offset points', ha='right')


ax.legend()
plt.title('t-SNE Visualization : x-vectors')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

w = 1519
h = 846

f.set_size_inches(w/96,h/96)
f.savefig('i_vectorsKNNtsne_eval.png', dpi=96)


# In[ ]:




