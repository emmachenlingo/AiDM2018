
"""
Created on Fri Sep 18 21:09:10 2015

@author: Wojtek Kowalczyk

This script demonstrates how implement the "global average rating" recommender 
and validate its accuracy with help of 5-fold cross-validation.

"""

import numpy as np
import sys
from time import time
before = time()

#load data
#ratings=read_data("ratings.dat")
ratings=[]
f = open("datasets/ratings.dat", 'r')
for line in f:
    data = line.split('::')
    ratings.append([int(z) for z in data[:3]])
f.close()
ratings=np.array(ratings)

"""
Alternatively, instead of reading data file line by line you could use the Numpy
genfromtxt() function. For example:

ratings = np.genfromtxt("ratings.dat", usecols=(0, 1, 2), delimiter='::', dtype='int')

will create an array with 3 columns.

Additionally, you may now save the rating matrix into a binary file 
and later reload it very quickly: study the np.save and np.load functions.
"""


#split data into 5 train and test folds
nfolds=5

#allocate memory for results:
rmse_train=np.zeros(nfolds)
rmse_test=np.zeros(nfolds)

mae_train=np.zeros(nfolds)
mae_test=np.zeros(nfolds)

#to make sure you are able to repeat results, set the random seed to something:
np.random.seed(111)

seqs=[x%nfolds for x in range(len(ratings))]
np.random.shuffle(seqs)

#for each fold:
for fold in range(nfolds):
    train_sel=np.array([x!=fold for x in seqs])
    test_sel=np.array([x==fold for x in seqs])
    train=ratings[train_sel]
    test=ratings[test_sel]

    #print(train.shape, test.shape)    
#calculate model parameters: mean rating over the training set:
    gmr=np.mean(train[:,2])

#calculate 
#apply the model to the train set:
    rmse_train[fold]=np.sqrt(np.mean((train[:,2]-gmr)**2))
    mae_train[fold]=np.mean(np.abs(train[:,2]-gmr))
#apply the model to the test set:
    rmse_test[fold]=np.sqrt(np.mean((test[:,2]-gmr)**2))
    mae_test[fold]=np.mean(np.abs(test[:,2]-gmr))
    
#print errors:
    print("Fold " + str(fold) + ": RMSE_train=" + str(rmse_train[fold]) + "; RMSE_test=" + str(rmse_test[fold]))
    print("Fold " + str(fold) + ": MAE_train=" + str(mae_train[fold]) + "; MAE_test=" + str(mae_test[fold]))

#print the final conclusion:
print("\n")
print("Root Mean Squared error on TRAIN: " + str(np.mean(rmse_train)))
print("Root Mean Squared error on  TEST: " + str(np.mean(rmse_test)))

print("Mean Absolute error on TRAIN: " + str(np.mean(mae_train)))
print("Mean Absolute error on  TEST: " + str(np.mean(mae_test)))
# Just in case you need linear regression: help(np.linalg.lstsq) will tell you 
# how to do it!
interval = time() - before
print("Time interval (s):{0}".format(interval))