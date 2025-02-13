
import pandas as pd
import numpy as np
from config import SIM_PARAM, rank, logger
from sklearn.preprocessing import OneHotEncoder
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
# get_purchase2()

def check_mul_paths(POSSIBLE_PATHS, filename):
    import pickle
    for path in POSSIBLE_PATHS:
        try:
            with open(path + filename, 'rb') as file:
                obj = pickle.load(file)
                return obj
        except FileNotFoundError:
            continue
    raise FileNotFoundError("File not found in all paths :(")

def getMNIST():
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    X = np.vstack((train_X, test_X))
    y = np.vstack((train_y, test_y))
    X = np.array(X).reshape(X.shape[0], 28*28)
    y = np.array(y)
    # TODO make all others not use one-hot-encoding
    return X, y, None

def get_purchase10(): # Author Jaap Meerhof
    import pickle
    # DATA_PATH = "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/acquire-valued-shoppers-challenge/"
    DATA_PATH = "/data/BioGrid/meerhofj/acquire-valued-shoppers-challenge/"
    # DATA_PATH = "/home/hacker/cloud_jaap_meerhof/SchoolCloud/Master Thesis/Database/acquire-valued-shoppers-challenge/"
    POSSIBLE_PATHS = ["/data/BioGrid/meerhofj/acquire-valued-shoppers-challenge/", \
                      "/home/hacker/cloud_jaap_meerhof/SchoolCloud/Master Thesis/Database/acquire-valued-shoppers-challenge/", \
                      "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/acquire-valued-shoppers-challenge/"]
    X = check_mul_paths(POSSIBLE_PATHS, "purchase_100_features.p")
    y = check_mul_paths(POSSIBLE_PATHS, "purchase_100_10_labels.p")

    # X = pickle.load(open(DATA_PATH+"purchase_100_features.p", "rb"))
    # y = pickle.load(open(DATA_PATH+"purchase_100_10_labels.p", "rb"))
    y = y.reshape(-1, 1)

    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder()
    encoder.fit(y)
    y = encoder.transform(y).toarray()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10_000, test_size=10_000, random_state=69)
    fName = []
    for i in range(600):
        fName.append(str(i))
    return X_train, y_train, X_test, y_test, fName

def get_purchase(amount_labels): # Author Jaap Meerhof
    import urllib.request
    import pickle
    print("> Downloading dataset \"purchase\" from JaapCloud1.0...")

    dataX = urllib.request.urlopen("https://jaapmeerhof.nl/index.php/s/5FWyosCLWJaXoHp/download").read() # X
    X = pickle.loads(dataX)
    url = None
    if   amount_labels == 2 :
        url = "https://jaapmeerhof.nl/index.php/s/QzyN4BTeaaiTX5e"
    elif amount_labels == 10 :
        url = "https://jaapmeerhof.nl/index.php/s/xYzz56jHQjMNZCn"
    elif amount_labels == 20:
        url = "https://jaapmeerhof.nl/index.php/s/ijpbJq2cbWQiSLf"
    elif amount_labels == 50:
        url = "https://jaapmeerhof.nl/index.php/s/gt2yr7ioAW9NMbi"
    elif amount_labels == 100:
        url = "https://jaapmeerhof.nl/index.php/s/AB8FrfGR4JXQLFi"
            
    datay = urllib.request.urlopen(url + "/download").read() # y
    y = pickle.loads(datay)
    print("> done downloading from JaapCloud1.0!")
    fName = []
    for i in range(101):
        fName.append(str(i))
    
    return X, y.squeeze(), fName.squeeze()

def get_texas():
    """Author: Jaap Meerhof
    """
    import pickle
    DATA_PATH = "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/texas/"
    # DATA_PATH = "/home/hacker/cloud_jaap_meerhof/SchoolCloud/Master Thesis/Database/texas/"
    X = pickle.load(open(DATA_PATH+"texas_100_v2_features.p", "rb"))
    y = pickle.load(open(DATA_PATH+"texas_100_v2_labels.p", "rb"))
    y = y.reshape((y.shape[0], 1))
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder()
    encoder.fit(y)
    y = encoder.transform(y).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10_000, test_size=10_000, random_state=69)
    fName = ['THCIC_ID', 'SEX_CODE', 'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION', \
             'LENGTH_OF_STAY', 'PAT_AGE', 'PAT_STATUS', 'RACE', 'ETHNICITY', \
                'TOTAL_CHARGES', 'ADMITTING_DIAGNOSIS'] # y= 'PRINC_SURG_PROC_CODE'
    return X_train, y_train, X_test, y_test, fName
# get_texas()
def get_iris():
    data = pd.read_csv('./dataset/iris.csv').values

    zero_index = data[:, -1] == 0
    one_index = data[:, -1] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]
    train_size_zero = int(zero_data.shape[0] * 0.8)
    train_size_one = int(one_data.shape[0] * 0.8)
    X_train, X_test = np.concatenate((zero_data[:train_size_zero, :-1], one_data[:train_size_one, :-1]), 0), \
                      np.concatenate((zero_data[train_size_zero:, :-1], one_data[train_size_one:, :-1]), 0)
    y_train, y_test = np.concatenate((zero_data[:train_size_zero, -1].reshape(-1,1), one_data[:train_size_one, -1].reshape(-1, 1)), 0), \
                      np.concatenate((zero_data[train_size_zero:, -1].reshape(-1, 1), one_data[train_size_one:, -1].reshape(-1, 1)), 0)

    fName = [['sepal length'],['sepal width'],['pedal length'],['pedal width']]

    
    
    return X_train, y_train, X_test, y_test, fName
# get_iris()

def get_give_me_credits():
    data = pd.read_csv('./dataset/GiveMeSomeCredit/cs-training.csv')
    data.dropna(inplace=True)
    fName = ['SeriousDlqin2yrs',
       'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']

    
    data = data[fName].values
    
    # Normalize the data
    data = data / data.max(axis=0)

    ratio = min(SIM_PARAM.N_SAMPLE / data.shape[0], 0.8)

    zero_index = data[:, 0] == 0
    one_index = data[:, 0] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]
    zero_ratio = len(zero_data) / data.shape[0]
    one_ratio = len(one_data) / data.shape[0]
    num = 10000
    train_size_zero = int(zero_data.shape[0] * ratio) + 1
    train_size_one = int(one_data.shape[0] * ratio)

    if rank == 1:
        print("Data Dsitribution")
        print(zero_ratio, one_ratio)

    X_train, X_test = np.concatenate((zero_data[:train_size_zero, 1:], one_data[:train_size_one, 1:]), 0), \
                      np.concatenate((zero_data[train_size_zero:train_size_zero+int(num * zero_ratio)+1, 1:], one_data[train_size_one:train_size_one+int(num * one_ratio), 1:]), 0)
    y_train, y_test = np.concatenate(
        (zero_data[:train_size_zero, 0].reshape(-1, 1), one_data[:train_size_one, 0].reshape(-1, 1)), 0), \
                      np.concatenate((zero_data[train_size_zero:train_size_zero+int(num * zero_ratio)+1, 0].reshape(-1, 1),
                                      one_data[train_size_one:train_size_one+int(num * one_ratio), 0].reshape(-1, 1)), 0)



    fNameNoLabel = ['RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']

    return X_train, y_train, X_test, y_test, fNameNoLabel

def get_default_credit_client():
    data = pd.read_csv('./dataset/DefaultsOfCreditCardsClient/UCI_Credit_Card.csv')
    data.dropna(inplace=True)

    fName = ["ID","LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
            "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
            "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
            "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6",
            "default.payment.next.month"]

    data = data[fName].values
    
    # Normalize the data
    data = data / data.max(axis=0)
    ratio = min(SIM_PARAM.N_SAMPLE / data.shape[0], 0.8)
    
    zero_index = data[:, -1] == 0
    one_index = data[:, -1] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]
    #trainSize = min(zero_data.shape[0], one_data.shape[0])
    train_size_zero = int(zero_data.shape[0] * ratio)
    train_size_one = int(one_data.shape[0] * ratio)
    if rank == 1:
        print("Data Dsitribution")
        print(train_size_zero, train_size_one)

    test_size_one = zero_data.shape[0] - train_size_zero 
    test_size_zero = one_data.shape[0] - train_size_one
    
    nTest = data.shape[0] * (1 - ratio)
    test_size = min(test_size_one, test_size_zero)

    X_train = np.concatenate((zero_data[:train_size_zero, :-1], one_data[:train_size_one, :-1]), 0)
                      
    X_test = np.concatenate((zero_data[train_size_zero: train_size_zero + test_size_one, :-1], 
                            one_data[train_size_one: train_size_one + test_size_one, :-1]), 0)
    
    y_train = np.concatenate((zero_data[:train_size_zero, -1].reshape(-1, 1), one_data[:train_size_one, -1].reshape(-1, 1)), 0)
    
    y_test = np.concatenate((zero_data[train_size_zero: train_size_zero + test_size_one, -1].reshape(-1, 1),
                            one_data[train_size_one: train_size_one + test_size_one, -1].reshape(-1, 1)), 0)


    fName = ["ID","LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
            "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
            "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
            "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]

    return X_train, y_train, X_test, y_test, fName

def get_adults():
    data = np.load('./dataset/adult.npy')
    data = data / data.max(axis=0)
    ratio = min(SIM_PARAM.N_SAMPLE / data.shape[0], 0.8)
    zero_index = data[:, 0] == 0
    one_index = data[:, 0] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]

    train_size_zero = int(zero_data.shape[0] * ratio)
    train_size_one = int(one_data.shape[0] * ratio)
    if rank == 1:
        print("Data Dsitribution")
        print(train_size_zero, train_size_one)

    test_size_one = zero_data.shape[0] - train_size_zero 
    test_size_zero = one_data.shape[0] - train_size_one

    # Use this to test training data with the same distribution
    # trainSize = min(zero_data.shape[0], one_data.shape[0])
    # train_size_zero = int(trainSize * 2 * ratio) + 1
    # train_size_one = int(trainSize * ratio)
    # if rank == 1:
    #     print("Data Dsitribution")
    #     print(train_size_zero, train_size_one)

    X_train, X_test = np.concatenate((zero_data[:train_size_zero, 1:], one_data[:train_size_one, 1:]), 0), \
                      np.concatenate((zero_data[train_size_zero:, 1:], one_data[train_size_one:, 1:]), 0)
    y_train, y_test = np.concatenate(
        (zero_data[:train_size_zero, 0].reshape(-1, 1), one_data[:train_size_one, 0].reshape(-1, 1)), 0), \
                      np.concatenate((zero_data[train_size_zero:, 0].reshape(-1, 1),
                                      one_data[train_size_one:, 0].reshape(-1, 1)), 0)

    segment_A = int(0.2 * (data.shape[1] - 1))
    segment_B = segment_A + int(0.2 * (data.shape[1] - 1))
    segment_C = segment_B + int(0.3 * (data.shape[1] - 1))

    return X_train, y_train, X_test, y_test, segment_A, segment_B, segment_C

    

def get_data_augment():
    data = np.random.rand(SIM_PARAM.N_SAMPLE * 2, SIM_PARAM.N_FEATURE)
    label = np.random.randint(2, size=SIM_PARAM.N_SAMPLE * 2)
    
    # Normalize the data
    data = data / data.max(axis=0)

    ratio = SIM_PARAM.N_SAMPLE / data.shape[0]

    zero_index = label == 0
    one_index = label == 1
    zero_data = data[zero_index]
    one_data = data[one_index]
    zero_label = label[zero_index]
    one_label = label[one_index]

    zero_ratio = len(zero_data) / data.shape[0]
    one_ratio = len(one_data) / data.shape[0]
    
    train_size_zero = int(zero_data.shape[0] * ratio)
    train_size_one = int(one_data.shape[0] * ratio)

    if rank == 1:
        print("Data Dsitribution")
        print(zero_ratio, one_ratio)

    X_train, X_test = np.concatenate((zero_data[:train_size_zero, 0:], one_data[:train_size_one, 0:]), 0), \
                      np.concatenate((zero_data[train_size_zero:, 0:], one_data[train_size_one:, 0:]), 0)
    y_train, y_test = np.concatenate((zero_label[:train_size_zero].reshape(-1, 1), one_label[:train_size_one].reshape(-1, 1)), 0), \
                      np.concatenate((zero_label[train_size_zero:].reshape(-1, 1), one_label[train_size_one:].reshape(-1, 1)), 0)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test, None