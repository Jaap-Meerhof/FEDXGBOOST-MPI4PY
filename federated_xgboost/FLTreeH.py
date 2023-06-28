import sys
import numpy as np
from common.BasicTypes import Direction
from config import  N_CLIENTS, logger, rank, comm
from mpi4py import MPI
from algo.LossFunction import LeastSquareLoss, LogLoss
from data_structure.TreeStructure import *
from data_structure.DataBaseStructure import *
from federated_xgboost.PerformanceLogger import CommunicationLogger,TimeLogger
from visualizer.TreeRender import FLVisNode
from copy import deepcopy

from sklearn import metrics
from federated_xgboost.XGBoostCommon import XgboostLearningParam, compute_splitting_score, SplittingInfo, FedDirRequestInfo, FedDirResponseInfo, PARTY_ID

class MSG_ID:
    TREE_UPDATE = 69
    RESPONSE_GRADIENTS = 70

class H_FLXGBoostClassifierBase():
    def __init__(self, treeSet):
        self.nTree = len(treeSet)
        self.trees: list(H_FLPlainXGBoostTree) = treeSet #self.assign_tree()
        
        self.dataBase = DataBase()
        self.label = []
        self.excTimeLogger = TimeLogger()

    def log_info(self):
        self.excTimeLogger.log()
        for tree in self.trees:
            tree.commLogger.log()

    def assign_tree():
        raise NotImplementedError

    def append_data(self, dataTable, fName = None):
        """
        Dimension definition: 
        -   dataTable   nxm: <n> users & <m> features
        -   name        mx1: <m> strings
        """
        self.dataBase = DataBase.data_matrix_to_database(dataTable, fName)
        logger.warning('Appended data feature %s to database of party %d', str(fName), rank)

    def append_label(self, labelVec):
        self.label = np.reshape(labelVec, (len(labelVec), 1))

    def print_info(self):
        featureListStr = '' 
        ret = self.dataBase.log()
        print(ret)

    def boost(self): # TODO MAKE for Horizondtal
        orgData = deepcopy(self.dataBase)
        y = self.label
        y_pred = np.zeros(np.shape(self.label))
        #y_pred = np.ones(np.shape(self.label)) * 0.5
        y_pred = np.ones(np.shape(self.label)) # TODO get initial prediction with probability of being in y
        
        # server creates first tree
        
        nprocs = comm.Get_size()
        """
        boosting PAX
        """
        # Start federated boosting
        tStartBoost = self.excTimeLogger.log_start_boosting()
        for i in range(-1, self.nTree):
            tStartTree = TimeLogger.tic()            

            if rank == PARTY_ID.SERVER:
                print(f"busy with tree {i} out of {self.nTree} so {(i*100/self.nTree)}% done")
                # server does prediction on send tree -> get predictions/gradients -> create new tree
                if i >= 1:
                    for partner in range(1, nprocs): # don't send first tree
                        comm.send(self.trees[i-1], dest=partner, tag=MSG_ID.TREE_UPDATE)
                
                gradientsandall = {}
                UnionGradients = np.array([])
                UnionHessians  = np.array([])

                for partner in range(1, nprocs): # recieve their computed gradients, hessians and others
                    gradientsandall[partner] = comm.recv(source=partner, tag=MSG_ID.RESPONSE_GRADIENTS)
                    UnionGradients = np.hstack((UnionGradients, gradientsandall[partner][0].squeeze()))
                    UnionHessians  = np.hstack((UnionHessians, gradientsandall[partner][1].squeeze())) 
                    
                pass # union of gradients
                
                dataFit = QuantiledDataBase(self.dataBase)
                pass # create new tree
                self.trees[i].fit(dataFit, UnionGradients, UnionHessians)

                self.excTimeLogger.log_dt_fit(tStartTree, treeID=i) # Log the executed time
                
            else: # other participants
                if i >= 0: # else we take y_pred from above 
                    tree: H_FLPlainXGBoostTree = comm.recv(source=PARTY_ID.SERVER, tag=MSG_ID.TREE_UPDATE)
                    self.trees[i] = tree # update your trees
                    y_pred = tree.predict(orgData)
                dataFit = QuantiledDataBase(self.dataBase)
                tree.get_gradients_hessians(y.squeeze(), y_pred.squeeze(), dataFit)
                g = dataFit.gradVec
                h = dataFit.hessVec
                # print(f"DEBUG g {g.shape}")
                # print(f"DEBUG h {h.shape}")
                # print(f"DEBUG labels {np.shape(self.label)}")
                # print(f"DEBUG y_pred = {y_pred.shape}")
                # print(f"DEBUG yshape = {y.shape}")


                comm.send((g,h), PARTY_ID.SERVER, tag=MSG_ID.RESPONSE_GRADIENTS)

        print("Received the abort boosting flag from AP")
        self.excTimeLogger.log_end_boosting(tStartBoost)

    def evaluateTree(self, yPred, y, treeid = int):
        newTreeGain = abs(self.trees[treeid].root.compute_score())
        loss = self.trees[treeid].learningParam.LOSS_FUNC.diff(y, yPred)
        print("Loss", abs(loss), "Tree Gain", newTreeGain)
        logger.warning("Boosting, TreeID: %d, Loss: %f, Gain: %f", treeid, abs(loss), abs(newTreeGain))
        return loss

    def evaluatePrediction(self, y_pred, y, treeid = None):
        y_pred = 1.0 / (1.0 + np.exp(-y_pred)) # Mapping to -1, 1
        y_pred_true = y_pred.copy()
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        result = y_pred - y
        acc = np.sum(result == 0) / y_pred.shape[0]
        auc = metrics.roc_auc_score(y, y_pred_true)
        logger.warning("Metrics, TreeID: %s, acc: %f, auc: %f", str(treeid), acc, auc)
        

        return acc, auc

    def predict(self, X, fName = None):
        y_pred = None
        data_num = X.shape[0]
        # Make predictions
        testDataBase = DataBase.data_matrix_to_database(X, fName)
        
        for treeID, tree in enumerate(self.trees):
            # Estimate gradient and update prediction
            logger.warning(f"PREDICTION id {treeID}")
            b = FLVisNode(tree.root)
            b.display(treeID)

            update_pred = tree.predict(testDataBase)
            if y_pred is None:
                y_pred = np.zeros_like(update_pred).reshape(data_num, -1)
            if rank == 0: # hier gaat ie fout
                update_pred = np.reshape(update_pred, (data_num, 1))
                y_pred += update_pred
        return y_pred
     

class H_PlainFedXGBoost(H_FLXGBoostClassifierBase):
    def __init__(self, nTree = 3):
        trees = []
        for i in range(nTree):
            tree = H_FLPlainXGBoostTree(i)
            trees.append(tree)
        super().__init__(trees)


class H_FLPlainXGBoostTree():
    def __init__(self, id, param:XgboostLearningParam = XgboostLearningParam()):
        self.learningParam = param
        self.root = FLTreeNode()
        self.nNode = 0
        self.treeID = id
        self.commLogger = CommunicationLogger(N_CLIENTS)

    def get_gradients_hessians(self, y, yPred, qDataBase: QuantiledDataBase):
        logger.info("Tree is growing column-wise. Current column: %d", self.treeID)

        """
        This function computes the gradient and the hessian vectors to perform the tree construction
        """
        # Compute the gradients and hessians
        if rank != PARTY_ID.SERVER:
            G = np.array(self.learningParam.LOSS_FUNC.gradient(y, yPred)).reshape(-1)
            # print(f"DEBUG: {G.shape}")
            #G = G/ np.linalg.norm(G)
            H = np.array(self.learningParam.LOSS_FUNC.hess(y, yPred)).reshape(-1)
            # print(f"DEBUG: {H.shape}")
            logger.debug("Computed Gradients and Hessians ")
            logger.debug("G {}".format(' '.join(map(str, G))))
            logger.debug("H {}".format(' '.join(map(str, H))))
            qDataBase.appendGradientsHessian(G, H)
    
    def optimal_split_finding(self, qDataBase: QuantiledDataBase) -> SplittingInfo:
        privateSM = qDataBase.get_merged_splitting_matrix()
        sInfo = SplittingInfo()
        if privateSM.size: # check it own candidate
            score, maxScore, bestSplitId = compute_splitting_score(privateSM, qDataBase.gradVec, qDataBase.hessVec, XgboostLearningParam.LAMBDA, XgboostLearningParam.GAMMA)
            if(maxScore > 0):
                sInfo.isValid = True
                sInfo.bestSplitParty = 0
                sInfo.selectedCandidate = bestSplitId
                sInfo.bestSplitScore = maxScore
        sInfo.log()
        
        if (sInfo.isValid):
            #print(rank, sInfo.get_str_split_info())
            sInfo = self.fed_finalize_optimal_finding(sInfo, qDataBase, privateSM)

        return sInfo

    def fed_finalize_optimal_finding(self, sInfo: SplittingInfo, qDataBase: QuantiledDataBase, privateSM = np.array([])):
        # Set the optimal split as the owner ID of the current tree node
        # If the selected party is me
        # TODO: Considers implement this generic --> direct in the grow method as post processing?
        if(rank == sInfo.bestSplitParty):
            sInfo.bestSplittingVector = privateSM[sInfo.selectedCandidate,:]
            feature, value = qDataBase.find_fId_and_scId(sInfo.bestSplittingVector)
                
            updateSInfo = deepcopy(sInfo)
            updateSInfo.bestSplittingVector = privateSM[sInfo.selectedCandidate,:]

            # Only the selected rank has these information so it saves for itself
            sInfo.featureName = feature
            sInfo.splitValue = value

        return sInfo

    def grow(self, qDataBase: QuantiledDataBase, depth, NodeDirection = TreeNodeType.ROOT, currentNode : FLTreeNode = None):
        logger.info("Tree is growing depth-wise. Current depth: {}".format(depth) + " Node's type: {}".format(NodeDirection))
        currentNode.nUsers = qDataBase.nUsers
        
        # Assign the unique fed tree id for each nodeand save the splitting info for each node
        currentNode.FID = self.nNode
        self.nNode += 1
        sInfo = self.optimal_split_finding(qDataBase) 
        # sInfo.log()
        currentNode.set_splitting_info(sInfo)
        if(sInfo.isValid):
            maxDepth = XgboostLearningParam.MAX_DEPTH
            # Construct the new tree if the gain is positive
            if (depth < maxDepth) and (sInfo.bestSplitScore > 0):
                depth += 1
                lD, rD = qDataBase.partition(sInfo.bestSplittingVector)
                logger.info("Splitting the database according to the best splitting vector.")
                logger.debug("\nOriginal database: %s", qDataBase.get_info_string())
                logger.debug("\nLeft splitted database: %s", lD.get_info_string())
                logger.debug("\nRight splitted database: %s \n", rD.get_info_string())

                # grow recursively
                currentNode.leftBranch = FLTreeNode()
                currentNode.rightBranch = FLTreeNode()
                self.grow(lD, depth,NodeDirection = TreeNodeType.LEFT, currentNode=currentNode.leftBranch)
                self.grow(rD, depth, NodeDirection = TreeNodeType.RIGHT, currentNode=currentNode.rightBranch)
            
            else:
                weight, score = FLTreeNode.compute_leaf_param(qDataBase.gradVec, qDataBase.hessVec, XgboostLearningParam.LAMBDA)
                currentNode.weight = weight
                currentNode.score = score
                currentNode.leftBranch = None
                currentNode.rightBranch = None

                logger.info("Reached max-depth or Gain is negative. Terminate the tree growing, generate the leaf with weight Leaf Weight: %f", currentNode.weight)
        else:
            weight, score = FLTreeNode.compute_leaf_param(qDataBase.gradVec, qDataBase.hessVec, XgboostLearningParam.LAMBDA)
            currentNode.weight = weight
            currentNode.score = score
            currentNode.leftBranch = None
            currentNode.rightBranch = None

            logger.info("Splitting candidate is not feasible. Terminate the tree growing and generate the leaf with weight Leaf Weight: %f", currentNode.weight)


    def fit(self, qDataBase: QuantiledDataBase, g, h):
        pass

        qDataBase.appendGradientsHessian(g, h) # set the gradients and hessians
        rootNode = FLTreeNode()
        self.grow(qDataBase, depth=1, NodeDirection= TreeNodeType.ROOT, currentNode = rootNode)
        self.root = rootNode

        b = FLVisNode(self.root)
        b.display(self.treeID)

    def predict(self, database:DataBase):
        curNode = self.root
        outputs = np.empty(database.nUsers, dtype=float)

        for userId in range(database.nUsers): # UserId unintuitively just means the different patients I guess
            while(not curNode.is_leaf()):
                direction = \
                    (database.featureDict[curNode.splittingInfo.featureName].data[userId] > curNode.splittingInfo.splitValue)
                if(direction == Direction.LEFT):
                    curNode =curNode.leftBranch
                elif(direction == Direction.RIGHT):
                    curNode = curNode.rightBranch
            outputs[userId] = curNode.weight
        return outputs