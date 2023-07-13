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
from concurrent.futures import ThreadPoolExecutor, as_completed
# from memory_profiler import profile

class MSG_ID:
    TREE_UPDATE = 69
    RESPONSE_GRADIENTS = 70

class H_FLXGBoostClassifierBase():
    def __init__(self, treeSet):
        self.nTree = len(treeSet[0])
        self.nClasses = len(treeSet)
        self.trees: np.array(np.array(H_FLPlainXGBoostTree)) = treeSet #self.assign_tree()
        
        self.dataBase = DataBase()
        self.label = []
        self.excTimeLogger = TimeLogger()

    def log_info(self): # TODO make usefull for multi-class
        self.excTimeLogger.log()
        for tree in self.trees:
            tree.commLogger.log()

    def assign_tree():
        raise NotImplementedError
    
    def set_qDataBase(self, qDataBase):
        self.qDataBase = qDataBase

    def append_data(self, dataTable, fName = None):
        """
        Dimension definition: 
        -   dataTable   nxm: <n> users & <m> features
        -   name        mx1: <m> strings
        """
        self.dataBase = DataBase.data_matrix_to_database(dataTable, fName)
        logger.warning('Appended data feature %s to database of party %d', str(fName), rank)

    def append_label(self, labelVec):
        # self.label = np.reshape(labelVec, (len(labelVec), 1))
        self.label = labelVec

    def print_info(self):
        featureListStr = '' 
        ret = self.dataBase.log()
        print(ret)

    def boost(self, initProbabilities): 
        self.initProbas = initProbabilities
        orgData = deepcopy(self.dataBase)
        y = self.label
        print(f"self.label={np.shape(y)}")
        # y_pred = np.zeros(np.shape(self.label))
        #y_pred = np.ones(np.shape(self.label)) * 0.5
        # y_pred = np.ones(np.shape(self.label)) # TODO get initial prediction with probability of being in y
        y_pred = np.tile(initProbabilities, (len(self.label) , 1)) #(Nclasses, xrows)
        
        # server creates first tree
        
        nprocs = comm.Get_size()
        """
        boosting PAX
        """
        # Start federated boosting
        tStartBoost = self.excTimeLogger.log_start_boosting()
        for i in range(-1, self.nTree):
            # for c in range(0, self.nClasses):
            tStartTree = TimeLogger.tic()            

            if rank == PARTY_ID.SERVER:
                print(f"busy with tree {i} out of {self.nTree} so {(i*100/self.nTree):.2f}% done")
                # server does prediction on send tree -> get predictions/gradients -> create new tree
                if i < self.nTree-1:
                    if i != -1:
                        for partner in range(1, nprocs): # don't send first tree
                            comm.send(self.trees[:, i], dest=partner, tag=MSG_ID.TREE_UPDATE)
                    
                    gradientsandall = {}
                    UnionGradients = np.array([[] for _ in range(self.nClasses)])
                    UnionHessians  = np.array([[] for _ in range(self.nClasses)])

                    for partner in range(1, nprocs): # recieve their computed gradients, hessians and others
                        gradientsandall[partner] = comm.recv(source=partner, tag=MSG_ID.RESPONSE_GRADIENTS)
                        print(f"UnionGradients{np.shape(UnionGradients)}, gradientsandall: {np.shape(gradientsandall[partner][0].T)}")
                        UnionGradients = np.hstack((UnionGradients, gradientsandall[partner][0].T))
                        UnionHessians  = np.hstack((UnionHessians, gradientsandall[partner][1].T)) 
                        
                    pass # TODO take actual additions of gradients on 
                    
                    # dataFit = QuantiledDataBase(self.dataBase)
                    dataFit = self.qDataBase
                    # with ThreadPoolExecutor(max_workers=10) as executor:
                    #     results = []
                    #     future_to_stuff = [executor.submit(H_FLPlainXGBoostTree.fit, tree, dataFit, UnionGradients[c, :], UnionHessians[c, :]) for c, tree in enumerate(self.trees[:, i+1])]
                    #     for future in future_to_stuff:
                    #         results.append(future.result())
                    #     self.trees[:, i+1] = results
                    self.trees[:, i+1] = np.array([tree.fit(dataFit, UnionGradients[c, :], UnionHessians[c, :]) for c, tree in enumerate(self.trees[:, i+1])])
        
                else: # send last tree
                    for partner in range(1, nprocs): # don't send first tree
                        comm.send(self.trees[:, i], dest=partner, tag=MSG_ID.TREE_UPDATE)
                self.excTimeLogger.log_dt_fit(tStartTree, treeID=i) # Log the executed time
                
            else: # other participants #this code should be illegal
                if i >= 0 and i < self.nTree - 1: # else we take y_pred from above 
                    trees: list(H_FLPlainXGBoostTree) = comm.recv(source=PARTY_ID.SERVER, tag=MSG_ID.TREE_UPDATE)
                    self.trees[:,i] = trees # update your trees
                    update_pred = np.array([tree.predict(orgData) for tree in trees]).T
                    print(np.shape(update_pred))
                    print(np.shape(y_pred))
                    # y_pred = y_pred.squeeze()
                    y_pred += update_pred
                elif i == self.nTree -1:
                    trees: H_FLPlainXGBoostTree = comm.recv(source=PARTY_ID.SERVER, tag=MSG_ID.TREE_UPDATE)
                    self.trees[:, i] = trees # update your trees
                    # Get last tree but don't send gradients
                    break
                    
                # dataFit = QuantiledDataBase(self.dataBase)
                # dataFit = self.qDataBase

                G = np.array(self.trees[0][0].learningParam.LOSS_FUNC.gradient(y, y_pred))#.reshape(-1)
                H = np.array(self.trees[0][0].learningParam.LOSS_FUNC.hess(y, y_pred))#.reshape(-1)
                # G = sum(overal waar de s > x > s )
                # G should not be tied to individual value, it should be tied to sum of that split

                # I should create a mapping Split candidate -> gradient, hessian

                # splits = self.qDataBase # QuantiledDataBase()
                # data = orgData

                # i want to send split, gradients and hessians summed to the splits
                

                comm.send((G,H), PARTY_ID.SERVER, tag=MSG_ID.RESPONSE_GRADIENTS)

        print("Received the abort boosting flag from AP")
        self.excTimeLogger.log_end_boosting(tStartBoost)

    def evaluateTree(self, yPred, y, treeid = int):
        newTreeGain = abs(self.trees[treeid].root.compute_score())
        loss = self.trees[treeid].learningParam.LOSS_FUNC.diff(y, yPred)
        print("Loss", abs(loss), "Tree Gain", newTreeGain)
        logger.warning("Boosting, TreeID: %d, Loss: %f, Gain: %f", treeid, abs(loss), abs(newTreeGain))
        return loss

    def evaluatePrediction(self, y_pred, y, treeid = None):
        # y_pred = np.argmax(y_pred, axis=1)
        print(y_pred[:10])
        print("---------")
        print(y[:10])
        y = np.argmax(y, axis=1)
        print("---------")
        print(y[:10])


        print(f"max = {np.max(y_pred)}, min ={np.min(y_pred)}")
        print(f"max = {np.max(y)}, min ={np.min(y)}")

        acc = np.sum(np.array(y_pred) == np.array(y))/y_pred.shape[0]
        # auc = metrics.roc_auc_score(y, y_pred.reshape(-1, 1))
        auc = 0.11
        logger.warning("Metrics, TreeID: %s, acc: %f, auc: %f", str(treeid), acc, auc)
        

        return acc, auc
    
    
    def predict_proba(self, X, fName, initProbability): # returns probabilities of all classes
        y_pred = self.predict(X, fName, initProbability) # returns weights
        for rowid in range(np.shape(y_pred)[0]):
            row = y_pred[rowid, :]
            wmax = max(row)
            wsum = 0
            for y in row: wsum += np.exp(y-wmax)
            y_pred[rowid, :] = np.exp(row-wmax) / wsum
        return y_pred



    def predict(self, X, fName, initProbability): # returns class number
        y_pred = self.predictweights(X, fName, initProbability) # get leaf node weights
        return np.argmax(y_pred, axis=1)
    
    def predictweights(self, X, fName = None, initProbability = None): # returns weights
        # y_pred = [None for n in range(self.nClasses)]
        y_pred = np.tile(initProbability, (len(X) , 1)) #(Nclasses, xrows)
        data_num = X.shape[0]
        # Make predictions
        testDataBase = DataBase.data_matrix_to_database(X, fName)
        print(f"DEBUG: {np.shape(X)}, {np.shape(y_pred)}, {np.shape(initProbability)}, {np.shape(self.trees)}")
        print(f"{initProbability}")
        for treeID in range(self.nTree):
            for c in range(self.nClasses):
                tree = self.trees[c][treeID]
                # Estimate gradient and update prediction
                logger.warning(f"PREDICTION id {treeID}")
                b = FLVisNode(tree.root)
                b.display(treeID)

                update_pred = tree.predict(testDataBase)
                if y_pred[c] is None:
                    y_pred[c] = initProbability[c] # TODO replace with initprobas
                if rank == 0: # hier gaat ie fout
                    # update_pred = np.reshape(update_pred, (data_num, 1))
                    # print(f"{np.shape(y_pred[:, c])}, {np.shape(update_pred)}")
                    y_pred[:, c] += update_pred
        return y_pred
     

class H_PlainFedXGBoost(H_FLXGBoostClassifierBase):
    def __init__(self, nTree = 3, nClasses= 3):
        self.nClasses = nClasses
        trees = [[] for n in range(nClasses)]
        for nClass in range(nClasses):
            trees[nClass] = [H_FLPlainXGBoostTree(i) for i in range(nTree)]
        trees = np.array(trees)
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
            # print(f"DEBUG: {G.shape}")
            #G = G/ np.linalg.norm(G)
            G = np.array(self.learningParam.LOSS_FUNC.gradient(y, yPred)).reshape(-1)
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
            sInfo = self.finalize_optimal_finding(sInfo, qDataBase, privateSM)

        return sInfo

    def finalize_optimal_finding(self, sInfo: SplittingInfo, qDataBase: QuantiledDataBase, privateSM = np.array([])):
        # Set the optimal split as the owner ID of the current tree node
        # If the selected party is me
        # TODO: Considers implement this generic --> direct in the grow method as post processing?
        if(rank == sInfo.bestSplitParty):
            sInfo.bestSplittingVector = privateSM[sInfo.selectedCandidate,:]
            feature, value = qDataBase.find_fId_and_scId(sInfo.bestSplittingVector)
            
            # updateSInfo = deepcopy(sInfo)
            # updateSInfo.bestSplittingVector = privateSM[sInfo.selectedCandidate,:]

            # Only the selected rank has these information so it saves for itself
            sInfo.featureName = feature
            sInfo.splitValue = value

        return sInfo
    # @profile
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
                sInfo.delSplittinVector()
                logger.info("Splitting the database according to the best splitting vector.")
                logger.debug("\nOriginal database: %s", qDataBase.get_info_string())
                logger.debug("\nLeft splitted database: %s", lD.get_info_string())
                logger.debug("\nRight splitted database: %s \n", rD.get_info_string())
                del qDataBase # memory fix?
                # del sInfo
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
        qDataBase.appendGradientsHessian(g, h) # set the gradients and hessians
        rootNode = FLTreeNode()
        self.grow(qDataBase, depth=1, NodeDirection= TreeNodeType.ROOT, currentNode = rootNode)
        self.root = rootNode

        b = FLVisNode(self.root)
        b.display(self.treeID)
        return self

    def predict(self, database:DataBase):
        curNode = self.root
        outputs = np.empty(database.nUsers, dtype=float)

        for userId in range(database.nUsers): # UserId unintuitively just means the different patients I guess
            curNode = self.root
            while(not curNode.is_leaf()):
                direction = \
                    (database.featureDict[curNode.splittingInfo.featureName].data[userId] > curNode.splittingInfo.splitValue)
                if(direction == Direction.LEFT):
                    curNode =curNode.leftBranch
                elif(direction == Direction.RIGHT):
                    curNode = curNode.rightBranch
            outputs[userId] = curNode.weight
        return outputs