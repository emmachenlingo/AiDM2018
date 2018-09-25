import numpy as np
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

Nmovie = np.max(ratings[:,1]).astype(int)
Nuser = np.max(ratings[:,0]).astype(int)

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
   
#calculate model parameters: mean rating over the training set:
    gmr=np.mean(train[:,2])

    train = train[np.argsort(train[:,1])]

    items = set(train[:,1])
    mar = np.zeros(Nmovie)

    for x in range(1,Nmovie+1):
        if x in items:
            mar[x-1] = np.mean(train[train[:,1]==x, 2])
        else:
            mar[x-1] = gmr

#calculate 
#apply the model to the train set:
    rmse_train[fold]=np.sqrt(np.mean((train[:,2]-mar[train[:,1]-1])**2))
    mae_train[fold]=np.mean(np.abs(train[:,2]-mar[train[:,1]-1]))
#apply the model to the test set:
    rmse_test[fold]=np.sqrt(np.mean((test[:,2]-mar[test[:,1]-1])**2))
    mae_test[fold]=np.mean(np.abs(test[:,2]-mar[test[:,1]-1]))
    
#print errors:
    print("Fold " + str(fold) + ": RMSE_train=" + str(rmse_train[fold]) + "; RMSE_test=" + str(rmse_test[fold]))
    print("Fold " + str(fold) + ": MAE_train=" + str(mae_train[fold]) + "; MAE_test=" + str(mae_test[fold]))

#print the final conclusion:
print("\n")
print("Root Mean Squared error on TRAIN: " + str(np.mean(rmse_train)))
print("Root Mean Squared error on  TEST: " + str(np.mean(rmse_test)))

print("Mean Absolute error on TRAIN: " + str(np.mean(mae_train)))
print("Mean Absolute error on  TEST: " + str(np.mean(mae_test)))

interval = time() - before
print("Time interval (s):{0}".format(interval))