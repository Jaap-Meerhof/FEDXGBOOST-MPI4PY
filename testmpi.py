from datetime import datetime
from pyexpat import model
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


if rank == 0:
    for partner in range(1, comm.Get_size()):
        print(f"waiting from message from {partner}")
        param = comm.recv(source= partner, tag = 1)
        print(param)
else: 
    # for partner in range(2, comm.Get_size()):
    if rank ==1:
        import time
        time.sleep(3)
    print(f"sending to {0}")
    comm.send(True, dest = 0, tag = 1)