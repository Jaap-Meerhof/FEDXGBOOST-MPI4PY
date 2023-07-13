import xgboost as xgb
import numpy as np

POSSIBLE_PATHS = ["/data/BioGrid/meerhofj/Database/", \
                      "/home/hacker/cloud_jaap_meerhof/SchoolCloud/Master Thesis/Database/", \
                      "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/"]


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
    # logger.warning(f"getting purchase {num} dataset!")

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
        # logger.warning(f"got purchase {num} dataset!")

        return X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow
        
    return returnfunc

X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getPurchase(10)()

xgboostmodel = xgb.XGBClassifier(max_depth=3, objective="multi:softmax",
                learning_rate=0.3, n_estimators=10, gamma=0.5, reg_alpha=1, reg_lambda=10)
xgboostmodel.fit(X_train, np.argmax(y_train, axis=1))
print('fitted!')