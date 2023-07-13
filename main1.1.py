POSSIBLE_PATHS = ["/data/BioGrid/meerhofj/Database/", \
                      "/home/hacker/cloud_jaap_meerhof/SchoolCloud/Master Thesis/Database/", \
                      "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/"]

from config import CONFIG, dataset, rank, logger, comm
import numpy as np


def check_mul_paths(filename):
    import pickle
    for path in POSSIBLE_PATHS:
        try:
            with open(path + filename, 'rb') as file:
                obj = pickle.load(file)
                return obj
        except FileNotFoundError:
            continue
    raise FileNotFoundError("File not found in all paths :(")

def makeOneHot(y):
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder()
    encoder.fit(y)
    y = encoder.transform(y).toarray()
    return y

def take_and_remove_items(arr, size): #sshoutout to Chat-gpt
    indices = np.random.choice(len(arr), size,replace=False )
    selected_items = np.take(arr, indices, axis=0)
    arr = np.delete(arr, indices, axis=0)
    return selected_items, arr

def getPurchase(num):
    #first try local
    # 
    logger.warning(f"getting purchase {num} dataset!")

    train_size = 10_000
    test_size = 10_000
    random_state = 69
    shadow_size = 30_000 # take in mind that this shadow_set is devided in 3 sets

    def returnfunc():
        X = check_mul_paths('acquire-valued-shoppers-challenge/' + 'purchase_100_features.p')
        y = check_mul_paths('acquire-valued-shoppers-challenge/' + 'purchase_100_' + str(num) + '_labels.p')
        total_size = shadow_size + test_size + train_size
        if not total_size < len(X) : raise Exception(f"your don't have enough data for these settings. your original X is of size {len(X)} ")
        
        X_shadow, X = take_and_remove_items(X, shadow_size)
        y = y.reshape(-1, 1)
        y = makeOneHot(y)
        y_shadow, y = take_and_remove_items(y, shadow_size)       
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=random_state)
        
        fName = []
        for i in range(600):
            fName.append(str(i))
        logger.warning(f"got purchase {num} dataset!")

        return X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow
        
    return returnfunc

def getTexas():
    logger.warning("getting Texas database!")
    train_size = 10_000
    test_size = 10_000
    random_state = 69
    shadow_size = 30_000    

    X = check_mul_paths('texas/' + 'texas_100_v2_features.p')
    y = check_mul_paths('texas/' + 'texas_100_v2_labels.p')
    fName = check_mul_paths('texas/' + 'texas_100_v2_feature_desc.p')

    total_size = shadow_size + test_size + train_size
    if not total_size < len(X) : raise Exception(f"your don't have enough data for these settings. your original X is of size {len(X)} ")
    
    X_shadow, X = take_and_remove_items(X, shadow_size)
    y = y.reshape(-1, 1)
    y = makeOneHot(y)
    y_shadow, y = take_and_remove_items(y, shadow_size)       
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=random_state)
    logger.warning("got Texas database!")

    return X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow

def getMNIST():
    return

def getSynthetic():
    return

def getCensus():
    return

def getDNA():
    return


dataset = 'purchase-10' 
CONFIG["dataset"] = dataset 
dataset_list = ['purchase-10', 'purchase-20', 'purchase-50', 'purchase-100', 'texas', 'MNIST', 'synthetic', 'Census', 'DNA']
get_databasefunc = {'purchase-10': getPurchase(10), 'purchase-20':getPurchase(20), 
                    'purchase-50':getPurchase(50), 'purchase-100':getPurchase(100), 
                    'texas':getTexas, 'MNIST':getMNIST, 'synthetic':getSynthetic, 
                    'Census':getCensus, 'DNA':getDNA
                   }[dataset]

# from data_preprocessing import *

from federated_xgboost.FLTreeHMulti import H_PlainFedXGBoost # USE HMULTI
from federated_xgboost.XGBoostCommon import XgboostLearningParam, PARTY_ID 
from data_structure.DataBaseStructure import QuantileParam
from algo.LossFunction import LeastSquareLoss, LogLoss, SoftMax


XgboostLearningParam.LOSS_FUNC = SoftMax()
XgboostLearningParam.LOSS_TERMINATE = 50 # this is not done right now in the multi-class approach!
XgboostLearningParam.GAMMA = CONFIG["gamma"]
XgboostLearningParam.LAMBDA = CONFIG["lambda"]
QuantileParam.epsilon = QuantileParam.epsilon
QuantileParam.thres_balance = 0.3

XgboostLearningParam.N_TREES = CONFIG["MAX_TREE"]
XgboostLearningParam.MAX_DEPTH = CONFIG["MAX_DEPTH"]

NCLASSES = 100
if CONFIG["model"] == "PlainXGBoost":
    model = H_PlainFedXGBoost(XgboostLearningParam.N_TREES, nClasses=NCLASSES)

logger.warning("TestInfo, {0}".format(CONFIG))
logger.warning("XGBoostParameter, nTree: %d, maxDepth: %d, lambda: %f, gamma: %f", 
XgboostLearningParam.N_TREES, XgboostLearningParam.MAX_DEPTH, XgboostLearningParam.LAMBDA, XgboostLearningParam.GAMMA)
logger.warning("QuantileParameter, eps: %f, thres: %f", QuantileParam.epsilon, QuantileParam.thres_balance)

def log_distribution(X_train, y_train, y_test):
    nTrain = len(y_train)
    nZeroTrain = np.count_nonzero(y_train == 0)
    rTrain = nZeroTrain/nTrain

    nTest = len(y_test)
    nZeroTest = np.count_nonzero(y_test == 0)
    rTest = nZeroTest / nTest
    logger.warning("DataDistribution, nTrain: %d, zeroRate: %f, nTest: %d, ratioTest: %f, nFeature: %d", 
    nTrain, rTrain, nTest, rTest, X_train.shape[1])

def test_global(model, getDatabaseFunc):
    X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getDatabaseFunc()
    log_distribution(X_train, y_train, y_test)
    model.append_data(X_train, fName)
    model.append_label(y_train)
    from data_structure.DataBaseStructure import QuantiledDataBase

    quantile = QuantiledDataBase(model.dataBase)

    initprobability = (sum(y_train))/len(y_train)

    total_users = comm.Get_size() - 1
    
    total_lenght = len(X_train)
    
    elements_per_node = total_lenght//total_users

    start_end = [(i * elements_per_node, (i+1)* elements_per_node) for i in range(total_users)]
    if rank != PARTY_ID.SERVER:
        start = start_end[rank-1][0]
        end = start_end[rank-1][1]

        X_train_my, X_test_my = X_train[start:end, :], X_test[start:end, :]
        y_train_my = y_train[start:end]


    # split up the database between the users
    if rank == PARTY_ID.SERVER:
        model.append_data(X_train, fName)
        model.set_qDataBase(quantile)
    else:
        quantile = quantile.splitupHorizontal(0, 5_000)
        model.set_qDataBase(quantile)
        model.append_data(X_train_my, fName)
        model.append_label(y_train_my)
    
    model.boost(initprobability)

    if rank == PARTY_ID.SERVER:
        y_pred = model.predict(X_test, fName, initprobability)
        
        import xgboost as xgb
        xgboostmodel = xgb.XGBClassifier(max_depth=3, objective="multi:softmax",
                            learning_rate=0.3, n_estimators=10, gamma=0.5, reg_alpha=1, reg_lambda=10)
        xgboostmodel.fit(X_train, np.argmax(y_train, axis=1))
        from sklearn.metrics import accuracy_score
        y_pred_xgb = xgboostmodel.predict(X_test)
        print(f"Accuracy xgboost normal = {accuracy_score(y_test, y_pred_xgb)}")
        print(y_pred)
    else:
        y_pred = [] # basically a none

    y_pred_org = y_pred.copy()
    X = X_train
    y = y_train 
    return X, y, y_pred_org, y_test, model, X_shadow, y_shadow


# main
if rank != -1:
    full_data = []
    # insert for loop here for multiple variables
    for n_trees in [5 , 10, 50, 100, 300]:
        CONFIG["MAX_DEPTH"] = n_trees
        paramname = "N_TREES"
        X, y, y_pred, y_test, model, X_shadow, y_shadow = test_global(model, get_databasefunc)
        if rank == PARTY_ID.SERVER:
            # model.log_info()
            import pickle
            pickle.dump({"model":model, "y_pred":y_pred, "y_test":y_test}, open( "debug.p", "wb"))
            # target_model = pickle.load(open(TARGET_MODEL_NAME, "rb"))

            acc, auc = model.evaluatePrediction(y_pred, y_test, treeid=99)    
            print("Prediction: ", acc, auc)
            from membership.MembershipInference import membership_inference
            from copy import deepcopy
            
            shadow_X = X.deepcopy()
            
            import xgboost as xgb
            shadowmodel = shadow_model = xgb.XGBClassifier(max_depth=CONFIG["MAX_TREE"], tree_method='approx', objective="multi:softmax", # "multi:softmax"
                                learning_rate=0.3, n_estimators=CONFIG["MAX_TREE"], gamma=CONFIG["gamma"], reg_alpha=0, reg_lambda=CONFIG["lambda"], min_child_weight = 1) 
            targetmodel = attack_model = xgb.XGBClassifier(tree_method="exact", objective='binary:logistic', max_depth=8, n_estimators=50, learning_rate=0.3) 

            data = membership_inference(X, y, X_shadow, y_shadow, model, shadow_model, attack_model)

            full_data.append(data)

            # from membership.plotting import prettyPlot
        
    labels = ["acc_training_target", "acc_test_target", "overfit_target", 
                "acc_training_shadow", "acc_test_shadow", "overfit_shadow", 
                "acc_X_attack", "acc_other_attack", 
                "precision_50_attack", "acc_50_attack"]
    labels = [paramname] + labels
    print(labels)
    print(full_data)
    import time
    ascii_time = time.strftime("%H:%M", time.localtime(time.time()))
    pickle.dump(data, open( f"fulldata/fulldata_{CONFIG['dataset']}_{ascii_time}.p", "wb")) # save the data for later plotting if needed

    # params = Params(N_TREES, MAX_DEPTH, ETA, REG_LAMBDA, REG_ALPHA, GAMMA, MIN_CHILD_WEIGHT, eA = EA, n_bins=N_BINS, n_participants=N_PARTICIPANTS, num_class=10)
    paramsstring = f""
    PLOT_NAME = "test.png"
    PLOT_TITLE = "testing"
    from membership.plotting import plot_data
    plot_data(np.array(full_data), labels, PLOT_NAME, paramsstring, suptext= PLOT_TITLE)
    # prettyPlot(fulldata)
