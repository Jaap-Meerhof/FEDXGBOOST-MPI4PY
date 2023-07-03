import numpy as np

class LeastSquareLoss:
    def diff(self, actual, predicted):
        return sum((actual - predicted)**2)

    def gradient(self, actual, predicted):
        return -(actual - predicted)

    def hess(self, actual, predicted):
        return np.ones_like(actual)

class LogLoss():
    def diff(self, actual, predicted):
        prob = 1.0 / (1.0 + np.exp(-predicted))
        return sum(actual * np.log(prob) + (1 - actual) * np.log(1 - prob))

    def gradient(self, actual, predicted):
        prob = 1.0 / (1.0 + np.exp(-predicted))
        return prob - actual

    def hess(self, actual, predicted):
        prob = 1.0 / (1.0 + np.exp(-predicted))
        return prob * (1.0 - prob) 
    
class SoftMax():
    def diff(self, actual, predicted):
        return 
    
    def gradient(self, actual, predicted): # TODO make this not incredibly redundant
        actualtmp = np.argmax(actual, axis=1)
        grad = np.zeros((predicted.shape), dtype=float) # for multi-class
        hess = np.zeros((predicted.shape), dtype=float) # for multi-class
        for rowid in range(predicted.shape[0]):
            wmax = max(predicted[rowid]) # line 100 multiclass_obj.cu
            wsum =0.0
            for i in predicted[rowid] : wsum +=  np.exp(i - wmax)
            for c in range(predicted.shape[1]):
                p = np.exp(predicted[rowid][c]- wmax) / wsum
                target = actualtmp[rowid]
                g = p - 1.0 if c == target else p
                h = max((2.0 * p * (1.0 - p)).item(), 1e-6)
                grad[rowid][c] = g
                hess[rowid][c] = h
        return np.array(grad)
    
    def hess(self, actual, predicted): # TODO make this not incredibly redundant
        actualtmp = np.argmax(actual, axis=1)
        grad = np.zeros((predicted.shape), dtype=float) # for multi-class
        hess = np.zeros((predicted.shape), dtype=float) # for multi-class
        for rowid in range(predicted.shape[0]):
            wmax = max(predicted[rowid]) # line 100 multiclass_obj.cu
            wsum =0.0
            for i in predicted[rowid] : wsum +=  np.exp(i - wmax)
            for c in range(predicted.shape[1]):
                p = np.exp(predicted[rowid][c]- wmax) / wsum
                target = actualtmp[rowid]
                g = p - 1.0 if c == target else p
                h = max((2.0 * p * (1.0 - p)).item(), 1e-6)
                grad[rowid][c] = g
                hess[rowid][c] = h
        return np.array(hess)