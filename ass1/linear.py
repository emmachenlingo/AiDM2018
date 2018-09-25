
import numpy as np
from time import time

#record time
before = time()

#load data
ratings = np.genfromtxt("datasets/ratings.dat", usecols=(0, 1, 2), delimiter='::', dtype='int')

#total number of movies and users
Nmovie = np.max(ratings[:,1]).astype(int)
Nuser = np.max(ratings[:,0]).astype(int)

#split data into 5 train and test folds
nfolds=5

#allocate memory for results. 'rmse' means root mean square error; 'mae' means mean absolute error
rmse_train=np.zeros(nfolds)
rmse_test=np.zeros(nfolds)

mae_train=np.zeros(nfolds)
mae_test=np.zeros(nfolds)

#to make sure the repeatability
np.random.seed(111)

seqs=[x%nfolds for x in range(len(ratings))]
np.random.shuffle(seqs)

for fold in range(nfolds):
    train_sel=np.array([x!=fold for x in seqs])
    test_sel=np.array([x==fold for x in seqs])
    train=ratings[train_sel]
    test=ratings[test_sel]

    #print(train.shape, test.shape)    
#calculate model parameters: mean rating over the training set:
    gmr=np.mean(train[:,2])

#R_user is the user averaged ratings
    train = train[np.argsort(train[:,0])]
    items = set(train[:,0])
    R_user = np.zeros(Nuser)

    for x in range(1,Nuser+1):
        if x in items:
            R_user[x-1] = np.mean(train[train[:,0]==x, 2])
        else:
            R_user[x-1] = gmr

#R_movie is the movie averaged ratings
    train = train[np.argsort(train[:,1])]
    items = set(train[:,1])
    R_movie = np.zeros(Nmovie)

    for x in range(1,Nmovie+1):
        if x in items:
            R_movie[x-1] = np.mean(train[train[:,1]==x, 2])
        else:
            R_movie[x-1] = gmr

#construct matrix for least square regression
    X = np.vstack([R_user[train[:,0]-1], R_movie[train[:,1]-1], gmr*np.ones(train.shape[0])]).T
    y = train[:,2]
    S = np.linalg.lstsq(X,y)

#prediction is the linear combination of the 3 kinds of averages    
    pred = np.dot(X, S[0][:,np.newaxis]).flatten()  

#prediction of the test data
    X_test = np.vstack([R_user[test[:,0]-1], R_movie[test[:,1]-1], gmr*np.ones(test.shape[0])]).T
    pred_test = np.dot(X_test, S[0][:,np.newaxis]).flatten()

#with rounding
    pred[pred>5] = 5
    pred[pred<1] = 1
    pred_test[pred_test>5] = 5
    pred_test[pred_test<1] = 1

#calculate 
#apply the model to the train set:
    rmse_train[fold]=np.sqrt(np.mean((train[:,2]-pred)**2))
    mae_train[fold]=np.mean(np.abs(train[:,2]-pred))
#apply the model to the test set:
    rmse_test[fold]=np.sqrt(np.mean((test[:,2]-pred_test)**2))
    mae_test[fold]=np.mean(np.abs(test[:,2]-pred_test))
    
#print errors:
    print("Fold " + str(fold) + ": RMSE_train=" + str(rmse_train[fold]) + "; RMSE_test=" + str(rmse_test[fold]))
    print("Fold " + str(fold) + ": MAE_train=" + str(mae_train[fold]) + "; MAE_test=" + str(mae_test[fold]))

#print the final conclusion:
print("\n")
print("Root Mean Squared error on TRAIN: " + str(np.mean(rmse_train)))
print("Root Mean Squared error on  TEST: " + str(np.mean(rmse_test)))

print("Mean Absolute error on TRAIN: " + str(np.mean(mae_train)))
print("Mean Absolute error on  TEST: " + str(np.mean(mae_test)))

#claculate the total run time
interval = time() - before
print("Time interval (s):{0}".format(interval))