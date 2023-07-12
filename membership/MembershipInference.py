import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

def split_shadowfake(shadow_fake):
    split = len(shadow_fake[0][:,0])//3 # 
    other_fake  = (shadow_fake[0][:split, :], shadow_fake[1][:split]) # splits the dataset
    test_fake   = (shadow_fake[0][split:2*split, :], shadow_fake[1][split:2*split])
    shadow_fake = (shadow_fake[0][2*split:, :], shadow_fake[1][2*split:]) # splits the datset
    return other_fake, test_fake, shadow_fake

def membership_inference(X, y, X_shadow, y_shadow, target_model, shadow_model, attack_model):
    other_fake, test_fake, shadow_fake = split_shadowfake((X_shadow, y_shadow))
    shadow_model.fit(shadow_fake[0], shadow_fake[1])
    y_pred = shadow_model.predict(test_fake[0])
    print("> shadow accuracy: %.2f" % (accuracy_score(test_fake[1], y_pred)))

    y0 = np.zeros(len(other_fake[0]))
    y1 = np.ones(len(shadow_fake[0]))

    x0 = other_fake[0]
    x1 = shadow_fake[0]
    x_shadow = np.concatenate((x0,x1), axis=0)
    y_shadow = np.concatenate((y0,y1), axis=0)
    indices = np.random.permutation(x_shadow.shape[0])
    x_shadow = x_shadow[indices]
    y_shadow = y_shadow[indices]


    attack_x_0 = shadow_model.predict_proba(np.array(x_shadow, dtype=float))
    tmp = attack_x_0
    attack_model.fit(tmp,y_shadow)

    test_x = np.vstack((X, test_fake[0]))
    y_attack = np.hstack( (np.ones(X.shape[0]), np.zeros(test_fake[0].shape[0])) )

    predicted = target_model.predict_proba(test_x)
    print(tmp.shape)
    print(predicted.shape)

    y_pred = attack_model.predict(predicted)
    print("> Attack accuracy: %.2f" % (accuracy_score(y_attack, y_pred)))
    print("> Attack precision: %.2f" % (precision_score(y_attack, y_pred)))

    data = []
    acc_training_target = accuracy_score(y, target_model.predict(X))
    data.append(acc_training_target)
    #   * accuracy target_model on test data
    acc_test_target = accuracy_score(shadow_fake[1], target_model.predict(shadow_fake[0])) # shadow_fake used for testing
    data.append(acc_test_target)
    #   * degree of overfitting
    overfit_target = acc_training_target - acc_test_target
    data.append(overfit_target)
    #   * accuracy shadow_model on training data
    acc_training_shadow = accuracy_score(shadow_fake[1], shadow_model.predict(shadow_fake[0]))
    data.append(acc_training_shadow)
    #   * accuracy shadow_model on test data
    acc_test_shadow = accuracy_score(test_fake[1], shadow_model.predict(test_fake[0]))
    data.append(acc_test_shadow)
    #   * degree of overfitting
    overfit_shadow = acc_training_shadow - acc_test_shadow
    data.append(overfit_shadow)
    #   * attack accuracy on X 

    acc_X_attack = accuracy_score(np.ones((X.shape[0],)), attack_model.predict(target_model.predict_proba(X)))
    data.append(acc_X_attack)
    #   * attack accuracy on other shadow
    acc_other_attack = accuracy_score(np.zeros((other_fake[1].shape[0],)), attack_model.predict(target_model.predict_proba(other_fake[0])))
    data.append(acc_other_attack)

    #   * precision of attack on both 50/50
    min = np.min((X.shape[0], other_fake[0].shape[0])) # takes the length of the shortest one
    fiftyfiftx = np.vstack((X[:min, :], other_fake[0][:min, :])) # TODO aggregate X and other_fake 50/50
    fiftyfifty = np.hstack((np.ones(min), np.zeros(min)))

    precision_50_attack = precision_score(fiftyfifty, attack_model.predict(target_model.predict_proba(fiftyfiftx)) )
    data.append(precision_50_attack)
    #   * accuracy both 50/50
    acc_50_attack = accuracy_score(fiftyfifty, attack_model.predict(target_model.predict_proba(fiftyfiftx)))
    data.append(acc_50_attack)
    # csv or json or pickle!
    return data