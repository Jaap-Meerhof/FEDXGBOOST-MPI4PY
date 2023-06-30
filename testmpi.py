from datetime import datetime
# from pyexpat import model
from statistics import mode
import mpi4py
import pandas as pd
import numpy as np
from algo.LossFunction import LeastSquareLoss, LogLoss
from data_structure.DataBaseStructure import QuantileParam
from federated_xgboost.FLTree import PlainFedXGBoost
from federated_xgboost.FedXGBoostTree import FEDXGBOOST_PARAMETER, FedXGBoostClassifier
from config import rank, logger, comm
from federated_xgboost.SecureBoostTree import PseudoSecureBoostClassifier, SecureBoostClassifier

from data_preprocessing import *
from federated_xgboost.XGBoostCommon import XgboostLearningParam, PARTY_ID 


# if rank == 0:
#     for partner in range(1, comm.Get_size()):
#         print(f"waiting from message from {partner}")
#         param = comm.recv(source= partner, tag = 1)
#         print(param)
# else: 
#     # for partner in range(2, comm.Get_size()):
#     if rank ==1:
#         import time
#         time.sleep(3)
#     print(f"sending to {0}")
#     comm.send(True, dest = 0, tag = 1)

def get_purchase2(): # Author Jaap Meerhof
    import pickle
    DATA_PATH = "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/acquire-valued-shoppers-challenge/"
    # DATA_PATH = "/home/hacker/cloud_jaap_meerhof/SchoolCloud/Master Thesis/Database/acquire-valued-shoppers-challenge/"
    X = pickle.load(open(DATA_PATH+"purchase_100_features.p", "rb"))
    y = pickle.load(open(DATA_PATH+"purchase_100_2_labels.p", "rb"))
    y = y.reshape((y.shape[0], 1))
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10_000, test_size=10_000, random_state=69)
    fName = []
    for i in range(600):
        fName.append(str(i))
    return X_train, y_train, X_test, y_test, fName
def log_distribution(X_train, y_train, y_test):
    nTrain = len(y_train)
    nZeroTrain = np.count_nonzero(y_train == 0)
    rTrain = nZeroTrain/nTrain

    nTest = len(y_test)
    nZeroTest = np.count_nonzero(y_test == 0)
    rTest = nZeroTest / nTest
    logger.warning("DataDistribution, nTrain: %d, zeroRate: %f, nTest: %d, ratioTest: %f, nFeature: %d", 
    nTrain, rTrain, nTest, rTest, X_train.shape[1])

import pickle

data = pickle.load(open("debug.p", "rb"))
datamain = pickle.load(open("debugmain.p", "rb"))

model, modelmain   = data["model"], datamain["model"]
y_pred, y_predmain = data["y_pred"], datamain["y_pred"]
y_test, y_predmain = data["y_test"], datamain["y_test"]

X_train, y_train, X_test, y_test, fName = get_purchase2()
log_distribution(X_train, y_train, y_test)
y_pred = model.predict(X_test, fName)
acc, auc = model.evaluatePrediction(y_pred, y_test, treeid=99)