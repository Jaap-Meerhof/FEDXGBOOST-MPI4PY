from datetime import datetime
from pyexpat import model
from statistics import mode
import mpi4py
import pandas as pd
import numpy as np
from algo.LossFunction import LeastSquareLoss, LogLoss
from data_structure.DataBaseStructure import QuantileParam
from federated_xgboost.FLTreeH import H_PlainFedXGBoost
from federated_xgboost.FedXGBoostTree import FEDXGBOOST_PARAMETER, FedXGBoostClassifier
from config import rank, logger, comm
from federated_xgboost.SecureBoostTree import PseudoSecureBoostClassifier, SecureBoostClassifier

from data_preprocessing import *
from federated_xgboost.XGBoostCommon import XgboostLearningParam, PARTY_ID 


from data_structure.DataBaseStructure import * # TMP for testing



def log_distribution(X_train, y_train, y_test):
    nTrain = len(y_train)
    nZeroTrain = np.count_nonzero(y_train == 0)
    rTrain = nZeroTrain/nTrain

    nTest = len(y_test)
    nZeroTest = np.count_nonzero(y_test == 0)
    rTest = nZeroTest / nTest
    logger.warning("DataDistribution, nTrain: %d, zeroRate: %f, nTest: %d, ratioTest: %f, nFeature: %d", 
    nTrain, rTrain, nTest, rTest, X_train.shape[1])

def test_purchase(model): # Author: Jaap Meerhof
    X_train, y_train, X_test, y_test, fName = get_purchase2()
    log_distribution(X_train, y_train, y_test)
    # model.append_data(X_train, fName)
    # model.append_label(y_train)
    # quantile = QuantiledDataBase(model.dataBase)
    # splits = quantile.get_merged_splitting_matrix()
    

    X_train_A, X_test_A = X_train[:5000, :], X_test[:5000, :]
    y_train_A = y_train[:5000]

    X_train_B, X_test_B = X_train[5000:, :], X_test[5000:, :]
    y_train_B = y_train[5000:]

    if rank == 1:
        model.append_data(X_train_A, fName)
        model.append_label(y_train_A)
    elif rank == 2:
        #print("Test", len(X_train_B), len(X_train_B[0]), len(y_train), len(y_train[0]))
        model.append_data(X_train_B, fName)
        model.append_label(y_train_B)
    elif rank == 0: #server
        model.append_data(X_train, fName)

    # model.print_info()
    model.boost()
    
    if rank == 1:
        y_pred = model.predict(X_test_A, fName)
        # print(model.predict_proba(X_test_A, fName))
    elif rank == 2:
        y_pred = model.predict(X_test_B, fName)
    else: # server
        pass
        y_pred = model.predict(X_test, fName)
        import xgboost as xgb
        xgboostmodel = xgb.XGBClassifier(max_depth=3, objective="binary:logistic",
                            learning_rate=0.3, n_estimators=3, gamma=0.5, reg_alpha=1, reg_lambda=10)
        xgboostmodel.fit(X_train, y_train)
        from sklearn.metrics import accuracy_score
        y_pred_xgb = xgboostmodel.predict(X_test)
        print(f"Accuracy xgboost normal = {accuracy_score(y_test, y_pred_xgb)}")

    y_pred_org = y_pred.copy()    

    return y_pred_org, y_test, model


def test_texas(model): # Author: Jaap Meerhof
    X_train, y_train, X_test, y_test, fName = get_texas()
    log_distribution(X_train, y_train, y_test)

    X_train_A, X_test_A = X_train[:, 0:5], X_test[:, 0:5]
    fNameA = fName[0:5]
    
    X_train_B, X_test_B = X_train[:, 5:11], X_test[:, 5:11]
    fNameB = fName[5:11]

    if rank == 1:
        model.append_data(X_train_A, fNameA)
        model.append_label(y_train)
    elif rank == 2:
        #print("Test", len(X_train_B), len(X_train_B[0]), len(y_train), len(y_train[0]))
        model.append_data(X_train_B, fNameB)
        model.append_label(np.zeros_like(y_train.shape))

    model.print_info()
    model.boost()
    
    if rank == 1:
        y_pred = model.predict(X_test_A, fNameA)
    elif rank == 2:
        y_pred = model.predict(X_test_B, fNameB)
    else:
        model.predict(np.zeros_like(X_test_A))

    y_pred_org = y_pred.copy()

    import xgboost as xgb
    xgboostmodel = xgb.XGBClassifier(max_depth=3, objective="binary:logistic",
                           learning_rate=0.3, n_estimators=3, gamma=0.5, reg_alpha=1, reg_lambda=10)
    xgboostmodel.fit(X_train, y_train)
    from sklearn.metrics import accuracy_score
    y_pred_xgb = xgboostmodel.predict(X_test)
    print(f"Accuracy xgboost normal = {accuracy_score(y_test, y_pred_xgb)}")

    return y_pred_org, y_test, model
    


def test_iris(model):
    X_train, y_train, X_test, y_test, fName = get_iris()
    log_distribution(X_train, y_train, y_test)

    X_train_A = X_train[:, 0].reshape(-1, 1)
    fNameA = fName[0]

    X_train_B = X_train[:, 2].reshape(-1, 1)
    fNameB = fName[2]
    X_train_C = X_train[:, 1].reshape(-1, 1)
    fNameC = fName[1]
    X_train_D = X_train[:, 3].reshape(-1, 1)
    fNameD = fName[3]
    X_test_A = X_test[:, 0].reshape(-1, 1)
    X_test_B = X_test[:, 2].reshape(-1, 1)
    X_test_C = X_test[:, 1].reshape(-1, 1)
    X_test_D = X_test[:, 3].reshape(-1, 1)

    if rank == 1:
        model.append_data(X_train_A, fNameA)
        model.append_label(y_train)
    elif rank == 2:
        #print("Test", len(X_train_B), len(X_train_B[0]), len(y_train), len(y_train[0]))
        model.append_data(X_train_B, fNameB)
        model.append_label(np.zeros_like(y_train))
    elif rank == 3:
        model.append_data(X_train_C, fNameC)
        model.append_label(np.zeros_like(y_train))
    elif rank == 4:
        model.append_data(X_train_D, fNameD)
        model.append_label(np.zeros_like(y_train))
    else:
        model.append_data(X_train_A)
        model.append_label(np.zeros_like(y_train))

    model.print_info()
    model.boost()
    
    if rank == 1:
        y_pred = model.predict(X_test_A, fNameA)
    elif rank == 2:
        y_pred = model.predict(X_test_B, fNameB)
    elif rank == 3:
        y_pred = model.predict(X_test_C, fNameC)
    elif rank == 4:
        y_pred = model.predict(X_test_D, fNameD)
    else:
        model.predict(np.zeros_like(X_test_A))

    y_pred_org = y_pred.copy()

    return y_pred_org, y_test, model

def test_aug_data(model):
    X_train, y_train, X_test, y_test, fName = get_data_augment()
    log_distribution(X_train, y_train, y_test)

    X_train_A = X_train[:, :2]
    X_test_A = X_test[:, :2]

    X_train_B = X_train[:, 2:]    
    X_test_B = X_test[:, 2:]
   
    if rank == 1:
        model.append_data(X_train_A)
        model.append_label(y_train)
    elif rank == 2:
        #print("Test", len(X_train_B), len(X_train_B[0]), len(y_train), len(y_train[0]))
        model.append_data(X_train_B)
        model.append_label(np.zeros_like(y_train))
    else:
        model.append_data(X_train_A)
        model.append_label(np.zeros_like(y_train))

    model.print_info()
    model.boost()
    
    if rank == 1:
        y_pred = model.predict(X_test_A)
    elif rank == 2:
        y_pred = model.predict(X_test_B)
    else:
        model.predict(np.zeros_like(X_test_A))

    y_pred_org = y_pred.copy()

    return y_pred_org, y_test, model

def test_give_me_credits(model):
    X_train, y_train, X_test, y_test, fName = get_give_me_credits()
    log_distribution(X_train, y_train, y_test)

    X_train_A = X_train[:, :2]
    fNameA = fName[:2]
    X_test_A = X_test[:, :2]

    #print(X_train_A)

    X_train_B = X_train[:, 2:]
    fNameB = fName[2:]
    X_test_B = X_test[:, 2:]

     # np.concatenate((X_train_A, y_train))
    if rank == 1:
        #print("Test A", len(X_train_A), len(X_train_A[0]), len(y_train), len(y_train[0]))
        #print("Test A", X_train_A.shape[0], len(X_train_A[0]), len(y_train), len(y_train[0]))
        model.append_data(X_train_A, fNameA)
        model.append_label(y_train)
    elif rank == 2:
        #print("Test", len(X_train_B), len(X_train_B[0]), len(y_train), len(y_train[0]))
        model.append_data(X_train_B, fNameB)
        model.append_label(np.zeros_like(y_train))
    else:
        model.append_data(X_train_A)
        model.append_label(np.zeros_like(y_train))


    model.print_info()


    model.boost()

    if rank == 1:
        y_pred = model.predict(X_test_A, fNameA)
    elif rank == 2:
        y_pred = model.predict(X_test_B, fNameB)
    else:
        model.predict(np.zeros_like(X_test_A))

    y_pred_org = y_pred.copy()
   
    return y_pred_org, y_test, model


def test_default_credit_client(model):
    X_train, y_train, X_test, y_test, fName = get_default_credit_client()
    log_distribution(X_train, y_train, y_test)

    X_train_A = X_train[:, 0:2]
    fNameA = fName[0:2]
    X_test_A = X_test[:, 0:2]

    X_train_B = X_train[:, 2:]
    fNameB = fName[2:]
    X_test_B = X_test[:, 2:]


    if rank == 1:
        model.append_data(X_train_A, fNameA)
        model.append_label(y_train)
    elif rank == 2:
        model.append_data(X_train_B, fNameB)
        model.append_label(np.zeros_like(y_train))
    else:
        model.append_data(X_train_A)
        model.append_label(np.zeros_like(y_train))


    model.print_info()
    model.boost()

    if rank == 1:
        y_pred = model.predict(X_test_A, fNameA)
    elif rank == 2:
        y_pred = model.predict(X_test_B, fNameB)
    else:
        model.predict(np.zeros_like(X_test_A))

    y_pred_org = y_pred.copy()
    
    return y_pred_org, y_test, model


def test_adult(model):

    X_train, y_train, X_test, y_test, segment_A, segment_B, segment_C = get_adults()
    log_distribution(X_train, y_train, y_test)
    
    X_train_A = X_train[:, 0:2]
    X_train_B = X_train[:, 2:]


    X_test_A = X_test[:, 0:2]
    X_test_B = X_test[:, 2:]

    if rank == 1:
        model.append_data(X_train_A)
        model.append_label(y_train)
    elif rank == 2:
        #print("Test", len(X_train_B), len(X_train_B[0]), len(y_train), len(y_train[0]))
        model.append_data(X_train_B)
        model.append_label(np.zeros_like(y_train))
    else:
        model.append_data(X_train_A)
        model.append_label(np.zeros_like(y_train))


    model.print_info()
    model.boost()

    if rank == 1:
        y_pred = model.predict(X_test_A)
    elif rank == 2:
        y_pred = model.predict(X_test_B)
    else:
        model.predict(np.zeros_like(X_test_A))

    y_pred_org = y_pred.copy()
    
    return y_pred_org, y_test, model

from sklearn import metrics
import sys
from config import CONFIG, dataset


def main():

    def pre_config():
        XgboostLearningParam.LOSS_FUNC = LogLoss()
        XgboostLearningParam.LOSS_TERMINATE = 50
        XgboostLearningParam.GAMMA = CONFIG["gamma"]
        XgboostLearningParam.LAMBDA = CONFIG["lambda"]
        QuantileParam.epsilon = QuantileParam.epsilon
        QuantileParam.thres_balance = 0.3

        XgboostLearningParam.N_TREES = CONFIG["MAX_TREE"]
        XgboostLearningParam.MAX_DEPTH = CONFIG["MAX_DEPTH"]

    try:
        pre_config()

        # Model selection
        if CONFIG["model"] == "PlainXGBoost":
            model = H_PlainFedXGBoost(XgboostLearningParam.N_TREES)
        elif CONFIG["model"] == "FedXGBoost":
            model = FedXGBoostClassifier(XgboostLearningParam.N_TREES)
        elif CONFIG["model"] == "SecureBoost": 
            model = SecureBoostClassifier(XgboostLearningParam.N_TREES)
        elif CONFIG["model"] == "PseudoSecureBoost":
            model = PseudoSecureBoostClassifier(XgboostLearningParam.N_TREES)


        # Log the test case and the parameters
        logger.warning("TestInfo, {0}".format(CONFIG))
        logger.warning("XGBoostParameter, nTree: %d, maxDepth: %d, lambda: %f, gamma: %f", 
        XgboostLearningParam.N_TREES, XgboostLearningParam.MAX_DEPTH, XgboostLearningParam.LAMBDA, XgboostLearningParam.GAMMA)
        logger.warning("QuantileParameter, eps: %f, thres: %f", QuantileParam.epsilon, QuantileParam.thres_balance)
        # Dataset selection  
        # y_pred, y_test, model = test_purchase(model) # TODO DELETE AFTER TEST
          
        if rank != -1:
            if CONFIG["dataset"] == dataset[0]:
                y_pred, y_test, model = test_iris(model)
            elif CONFIG["dataset"] == dataset[1]:
                y_pred, y_test, model = test_give_me_credits(model)
            elif CONFIG["dataset"] == dataset[2]:
                y_pred, y_test, model = test_adult(model)
            elif CONFIG["dataset"] == dataset[3]:
                y_pred, y_test, model = test_default_credit_client(model)
            elif CONFIG["dataset"] == dataset[4]:
                y_pred, y_test, model = test_aug_data(model)
            elif CONFIG["dataset"] == dataset[5]:
                y_pred, y_test, model = test_texas(model) # TODO make
            elif CONFIG["dataset"] == dataset[6]:
                y_pred, y_test, model = test_purchase(model)
            if rank == PARTY_ID.SERVER:
                model.log_info()
                acc, auc = model.evaluatePrediction(y_pred, y_test, treeid=99)    
                print("Prediction: ", acc, auc)
    

    except Exception as e:
        logger.error("Exception occurred", exc_info=True)
        print("Rank ", rank, e)




main()

def automated():
    """
    Currently not using this because the data synchronization is not yet verified.
    """
    modelArr = ["PlainXGBoost", "FedXGBoost", "SecureBoost", "PseudoSecureBoost"]
    dataset = ["Iris", "GiveMeCredits", "Adult", "DefaultCredits"]

    try:
        for i in range (len(modelArr)):
            CONFIG["model"] = modelArr[i]
            for j in range(len(dataset)):
                CONFIG["dataset"] = dataset[j]
                print("Rank", rank, "Testing", CONFIG["model"], CONFIG["dataset"])

    except Exception as e:
            logger.error("Exception occurred", exc_info=True)
            print("Rank ", rank, e)
        