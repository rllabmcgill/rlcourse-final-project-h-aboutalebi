import time
import numpy as np
import random
from sklearn.ensemble import ExtraTreesRegressor
from NewExperimentDenoingAlgorithm import *


#0 stands for red observation, 1 stands for blue observation, 2 stands for green observation
#actions can be performed in this state [0,1,2,3] corresponding to [left,right,up, down]
#options are [0,1,2,3] standing for "Just Taking": [left,right,up, down]
#observation are [0,1] which stands for [terminal,continue]
class ConstructEnvironment:
    ClassListofequal=[]
#BiasIndex determine the distance between each two column in the torus. n determines the number of nodes.(The true number of nodes is n**2)
    def __init__(self, gamma,stepSize, lambda1,alpha, numIteration,FileName, rank,Number_Columns=12, Number_Rows=10,startingCell=[0,0],Goal=[9,11]):
        np.set_printoptions(threshold=np.nan)
        self.Grid=[]
        self.Number_Columns=Number_Columns
        self.Number_Rows=Number_Rows
        self.blocks=set()
        self.gamma=gamma
        self.R1=0
        self.options=[0,1,2,3]
        self.optionProbabilityHenkel=[0.25,0.25,0.25,0.25] # list for probability of taking option in learning Henkel Matrix
        self.observations=[0,1]
        self.rewards=np.zeros((self.Number_Rows,self.Number_Columns))
        self.startingCell=startingCell
        self.GoalCell=Goal
        self.FileName = FileName
        self.lambda1 = lambda1
        self.rank=rank
        self.NumberOfZero=0
        self.alpha=alpha #Third constraint factor
        self.Ratio_Zero=0
        self.TrainingSet=[]
        self.Weight=np.zeros(10)
        self.PTH = {}  # PTH is a dictionary of all joint probabilty of test and histories for without Denoing  ***used for dynamic programming***
        self.ConditionalPTH = {}  # the same as PTH but instead restore conditional probability of P(Y_i|Y_1,...)
        self.indexMap = {}  # used for finiding the index of a given element inside a list(in B_TaoH is used) for History
        self.DenoisingPTH = {}  # DenoinsingPTH is a dictionary of all joint probabilty of test and histories for Denoing Method
        self.Without_DenoisingPTH={} # DenoinsingPTH is a dictionary of all joint probabilty of test and histories for Non-Denoing Method
        self.ExtendedWeight = []
        self.FinalEqualSum=[] #It stores the third constraint terms
        self.MapForListofEqualEntitis={} # store the map for equal elements **for speed up function Listofequalities
        self.MapForCounting={} #stores a map for number of time a sequence action observation has occured
        self.create()
        self.explore()
        self.Exam(stepSize, numIteration)
        # if (type(self.R1) == bool):
        #     if (self.R1 == False):
        #         with open(self.FileName, "a") as myfile:
        #             myfile.write("**********Experiment Repeated********************Experiment Repeated********************Experiment Repeated********************Experiment Repeated**********")
        #         A = ConstructEnvironment(TrainingData=self.TrainingSet,alpha=self.alpha, rank=self.rank,TestData=TestData,stepSize=self.R2/5, numIteration=numIteration, FileName=self.FileName, lambda1=self.lambda1)
        #         self.R1, self.R2,self.R1_train,self.R2_train =A.R1, A.R2,A.R1_train,A.R2_train
        #         self.lambda1=A.lambda1
        # with open(self.FileName, "a") as myfile:
        #     myfile.write("Size Of Training: " + str(len(TrainingData)) + '\n')
        #     myfile.write("Rank: " + str(self.rank) + '\n')
        #     myfile.write("stepsize: " + str(stepSize) + " lambda1: " + str(lambda1) + '\n')
        #     myfile.write("Average error Method without Denoising: " + str(self.R1) + '\n')
        #     myfile.write("Average error Method with Denoising: " + str(self.R2) + '\n')
        # print("Size Of Training: ", len(TrainingData))
        # print("Average error Method without Denoising", self.R1)
        # print("Average error Method with Denoising", self.R2)
        # print("stepsize: ", str(stepSize) , " lambda1: " , str(lambda1) , '\n')

# creates and draw the grid environment. we draw the figure patch by patch
    def create(self):
        B0=[]
        for i in range(self.Number_Rows):
            for j in range(self.Number_Columns):
                self.Grid.append([i, j])
                if(i==0):
                    B0.append([-1,j])
                if(j==0):
                    B0.append([i, -1])
                if(i==self.Number_Rows-1):
                    B0.append([self.Number_Rows, j])
                if (j==self.Number_Columns - 1):
                    B0.append([i, self.Number_Columns])
        B1=[]
        for i in range(5):
            B1.append([i,9])
        B2 = []
        for i in range(6):
            B2.append([5, i])
        B3=[[8,4],[9,4]]
        B=B0+B1+B2+B3
        for element in B:
            self.blocks.add(self.unpack(element))
        self.rewards[self.GoalCell[0],self.GoalCell[1]]=100

#Implements our policy
    def greedy(self,b1,ExtraTrees,Q, R, NewP_THR, InverseR, Newb_inf,Flag_Denoising=False):
        state=b1
        observation=0
        current_position=self.startingCell
        option=self.Max_option(ExtraTrees,state)
        reward=0
        k=0
        counter=1
        max_iteration=0
        while(current_position!=self.GoalCell and max_iteration<100):
            Flag=False
            New_position=self.Move(current_position,option)
            print("step ", k,": ",New_position)
            k+=1
            if (self.unpack(New_position) == self.unpack([-1, -1])):
                Flag = True
                observation = 0
                reward-=counter*10
            else:
                observation=1
            if(Flag_Denoising==False):
                state=self.State_Update_WithOutDenoising(TEST=[option,observation],O=Q,P_THR=NewP_THR,R=R,b_inf=Newb_inf,belief_state=state)
            else:
                state = self.State_Update_WithDenoising(TEST=[option, observation], Q=Q, R=R, NewP_THR=NewP_THR,InverseR=InverseR, Newb_inf=Newb_inf, Newb1=state)
            print(reward)
            if(Flag==False):
                current_position=New_position
                reward += counter*self.rewards[New_position[0], New_position[1]]
            else:
                option = self.Max_option(ExtraTrees, state)
            counter=counter*self.gamma
            max_iteration+=1
        return reward



# explores the environment to gather trajectories for constructing Henkel matrix and learning policy
    def explore(self):
        i=1
        observation=-1
        traning=[]
        Flag=False
        current_position=self.startingCell
        option=self.getAction()
        while(i<=1000):
            new_position=self.Move(current_position,option)
            if(self.unpack(new_position)==self.unpack([-1,-1])):
                Flag=True
                observation=0
            else:
                observation=1
            traning.append(option)
            traning.append(observation)
            if(Flag==False):
                current_position=new_position
            if(Flag):
                option=self.getAction()
                Flag=False
            if(i%5==0):
                self.TrainingSet.append(traning)
                traning=[]
            i+=1
        # print(self.TrainingSet)


# explores the environment to gather trajectories for constructing Henkel matrix and learning policy
    def explore_robot(self,training_size,b1,Q, R, NewP_THR, InverseR, Newb_inf,Flag_Denoising):
        i=1
        t1=copy.deepcopy(b1)
        observation=-1
        Flag=False
        current_position=self.startingCell
        option=self.getAction()
        Batch=[]
        while(i<=training_size):
            traning = []
            new_position=self.Move(current_position,option)
            reward=self.rewards[new_position[0],new_position[1]]
            if(self.unpack(new_position)==self.unpack([-1,-1])):
                Flag=True
                observation=0
                reward=-10
            else:
                observation=1
            traning.append(option)
            traning.append(observation)
            if(Flag_Denoising):
                b2=self.State_Update_WithDenoising( traning, Q, R, NewP_THR, InverseR, Newb_inf, b1)
            else:
                b2 = self.State_Update_WithOutDenoising(traning, Q, NewP_THR, R, Newb_inf, b1)
            # if(current_position==self.GoalCell):
            #     reward=100000
            if(new_position==self.GoalCell):
                reward=0
            elif(Flag==False):
                reward=-1
            else:
                reward=-11
            Batch.append([b1,option,reward,b2])
            b1=b2
            if(Flag==False):
                current_position=new_position
                if(current_position==self.GoalCell):
                    current_position=self.startingCell
                    b1=t1
                    option = self.getAction()
            if(Flag):
                option = self.getAction()
                Flag=False
            i+=1
        return Batch

#Implements Fitted_Q_Iteration
    def Fitted_Q_Iteration(self,Batch,NumberOfIteration,gamma):
        X = []
        extra_tree=ExtraTreesRegressor(max_depth=5)
        for i in range(len(Batch)):
            S1=[]
            S2=[]
            for j in range(Batch[i][0].shape[1]):
                S1.append(Batch[i][0][0,j])
                S2.append(Batch[i][3][0, j])
            Batch[i][0]=S1
            Batch[i][3] = S2
        for i in range(len(Batch)):
            X.append(Batch[i][0]+[Batch[i][1]])
        for i in range(NumberOfIteration):
            print("step: ",str(i))
            Y=np.zeros(len(Batch))
            for j in range(len(Batch)):
                if(i==0):
                    Y[j]=Batch[j][2]
                else:
                    Max=-1000000000000000
                    for k in self.options:
                        p=extra_tree.predict([Batch[j][3]+[k]])[0]
                        if(p>Max):
                            Max=p
                    Y[j]=Batch[j][2]+gamma*Max
                    # if(Batch[j][2]==100000):
                    #     Y[j]=100
            extra_tree.fit(np.array(X),np.array(Y))
        return extra_tree





    def Exam(self,stepSize, numIteration):
        O=self.constructDynamicMatriices(self.TrainingSet)
        # print(O[0])
        self.MapForCounting.clear()
        self.PTH.clear()
        self.ConditionalPTH.clear()
        O_Modified = O[0]
        NumberOfHelkenColumn = self.TESTelemet.__len__()
        HenkelMatrix = O[0][:, 0:NumberOfHelkenColumn]
        U, S, R = np.linalg.svd(HenkelMatrix)
        # print("Singular Values: ", S)
        Helper = []
        RANK=self.rank
        for singular in S:
            Helper.append(singular)
        with open(self.FileName, "a") as myfile:
            myfile.write('\n' + "SingularValues: " + str(Helper) + '\n')
        R = R[0:RANK, :]
        b1 = HenkelMatrix[0, :] * R.T
        P_THR = np.linalg.pinv(HenkelMatrix * R.T)
        b_inf = P_THR * (HenkelMatrix[:, 0])
        Batch_WithoutDenoising=self.explore_robot(training_size=800,b1=b1,Q=O,R=R,NewP_THR=P_THR,InverseR=0,Newb_inf=b_inf,Flag_Denoising=False)
        E=self.Fitted_Q_Iteration( Batch_WithoutDenoising, 4, gamma=self.gamma)
        q=self.unpack_vector(b1)
        # print(E.predict(np.matrix(np.hstack((q,[0])))))
        # print(E.predict(np.matrix(np.hstack((q,[1])))))
        # print(E.predict(np.matrix(np.hstack((q,[2])))))
        # print(E.predict(np.matrix(np.hstack((q,[3])))))
        # print("IMPORTANT .....")
        self.R1=self.greedy(b1=b1,ExtraTrees=E,Q=O,R=R,NewP_THR=P_THR,InverseR=[],Newb_inf=b_inf)

        # W = newAlgorithm(rank=5, DynamicMatrix=O_Modified, weights=self.ExtendedWeight,
        #                  ListEqualEntitities=ConstructEnvironment.ClassListofequal,
        #                  ListOfSumEqualEntities=self.FinalEqualSum, lambda1=self.lambda1, step=stepSize,
        #                  numberOfStep=numIteration,
        #                  FileName=self.FileName,alpha=self.alpha)  # the hyperparameter of second constarints are adjusted here
        # # self.environment.ExtendedWeight.clear()
        # Q, R = W.GradientEngine()
        # if (type(Q) == bool):
        #     if (Q == False):
        #         self.R1, self.R2 = False, R
        #         return
        # InverseR = R.T
        # InverseR = InverseR[:NumberOfHelkenColumn, :]
        # NewP_TH = Q * R[:, :NumberOfHelkenColumn]
        # Newb1 = (NewP_TH[0]) * InverseR
        # NewP_THR = np.linalg.pinv(NewP_TH * InverseR)
        # Newb_inf = NewP_THR * (NewP_TH[:, 0])

    # computes and returns an array containing all the observable matrices inculiding P(H,T),B(H,ao,T) for all ao
    def constructDynamicMatriices(self, TrainingSet):
        self.HISTORYelement = self.combination()
        self.TESTelemet = self.combination(Flag=True)
        self.ExtendedTest=self.combinationExtended()
        # print(T)
        self.Weight = np.zeros((self.HISTORYelement.__len__(), self.ExtendedTest.__len__()))
        O1 = []
        tip=0
        MAXLEN=-10
        q=0
        for h in self.HISTORYelement:
                # print(q)
                q+=1
                for t in self.ExtendedTest:
                    self.MapForCounting[self.unpack(h+t)]=0
        for h in self.HISTORYelement:
                for t in self.ExtendedTest:
                    if(len(h+t)>MAXLEN):
                        MAXLEN=len(h+t)
                    if(len(h+t)!=0):
                        self.MapForCounting[self.unpack((h+t)[:len(h+t)-1])]=0
        for data in self.TrainingSet:
            z=[]
            for k in data:
                z.append(k)
                if(len(z)>MAXLEN):
                    break
                self.MapForCounting[self.unpack(z)]+=1
        print("Hooray")
        for history in self.HISTORYelement:
            l = []
            tip+=1
            for test in self.ExtendedTest:
                Output=self.ProbabilityTestGivenHistory(test, history, self.indexMap[self.unpack(history)], self.indexMap[self.unpack(test)], TrainingSet)
                l.append(Output[1])
                self.Weight[self.indexMap[self.unpack(history)],self.indexMap[self.unpack(test)]]=Output[0]
            O1.append(l)
        self.ExtendedWeight = np.matrix(self.Weight)
        O2 = {}
        O1 = np.matrix(O1)
        print("Hooray")
        for action in self.options:
            for observation in self.observations:
                O3 = []
                l=np.matrix(O1[:,self.indexMap[self.unpack([action,observation])]])
                for test in self.TESTelemet:
                    if (test.__len__()!=0):
                        l = np.hstack((l,O1[:,self.indexMap[self.unpack([action,observation]+test)]]))
                O2[(action, observation)] = l
        R = []
        R.append(O1)
        R.append(O2)
        return R


    # calculates the joint probability of a test given its history and its corresponding weights P(A_1,Y_1,A_2,Y_2, ... ,Y_n) = P(Y_1|A_1)*P(Y_2|A_2,A_1,Y_1) *** it is implemented via dynamic programming for faster computational time ***
    def ProbabilityTestGivenHistory(self, Test, History, indexRow, indexColumn, TrainingSet):
        TEST = History + Test
        if self.unpack(TEST) in self.PTH:
            self.PTH[self.unpack(TEST)]=[self.PTH[self.unpack(TEST)][0],self.PTH[self.unpack(TEST)][1]]
            return (self.PTH[self.unpack(TEST)][0],self.PTH[self.unpack(TEST)][1])
        if (TEST.__len__() == 0):
            self.PTH[self.unpack(TEST)] = [1,1]
            return (1,1)
        ProbabilityOfTest=1
        for i in range(1,TEST.__len__(),2):
            p=self.ComplementaryProbabilty(TEST[i],TEST[:i],TrainingSet)
            if(p==0):
                # self.PTH[self.unpack(TEST)] = [0, 0]
                # return (1,0)
                ProbabilityOfTest=0
                break
            ProbabilityOfTest=ProbabilityOfTest*p
        # Old vesrion
        # HistoryCounter = 0
        # for o in TrainingSet:
        #     if self.TwoGivenListIsEqual(TEST[0::2], o[0::2]):
        #         HistoryCounter += 1
        # New vesrion:
        HistoryCounter = self.MapForCounting[self.unpack(TEST[:len(TEST) - 1])]
        if (HistoryCounter == 0):
            self.PTH[self.unpack(TEST)] = [0,0]
            return (0,0)
        weight=HistoryCounter #this is the best weight. Do Not Change it!
        self.PTH[self.unpack(TEST)] = [weight,ProbabilityOfTest]
        return (weight, ProbabilityOfTest)

#Update the current state for WithOutDenoising alg
    def State_Update_WithOutDenoising(self, TEST, O, P_THR, R, b_inf, belief_state):
        Sum = P_THR*O[1][(TEST[0], TEST[1])]*R.T
        return (belief_state * Sum) /((belief_state * Sum*b_inf)[0,0])

# Update the current state for WithDenoising alg
    def State_Update_WithDenoising(self, TEST, Q, R, NewP_THR, InverseR, Newb_inf, Newb1):
        New_R = self.indexMatrix([TEST[0], TEST[1]], R)
        NewP_TaoH = Q * New_R
        Sum = NewP_THR * NewP_TaoH *InverseR
        return (Newb1 * Sum) / ((Newb1*Sum*Newb_inf)[0,0])

# This is a complementray function for def ProbabilityTestGivenHistory. It calculates P(Y_i|Y_1,A_1,Y_2, ..., A_i-1)
    def ComplementaryProbabilty(self,y,history,TrainingSet):
        Test=history+[y]
        if (self.unpack(Test) in self.ConditionalPTH):
            return self.ConditionalPTH[self.unpack(Test)]
        else:
            # old version:
            # HistoryCounter = 0
            # HistoryDenumenatorCounter = 0
            # for data in TrainingSet:
            #     if(self.TwoGivenListIsEqual(Test,data)):
            #         HistoryCounter+=1
            #     if(self.TwoGivenListIsEqual(history,data)):
            #         HistoryDenumenatorCounter+=1
            # new version:
            HistoryCounter=self.MapForCounting[self.unpack(Test)]
            HistoryDenumenatorCounter=self.MapForCounting[self.unpack(history)]
            if (HistoryDenumenatorCounter == 0):
                self.ConditionalPTH[self.unpack(Test)]=0
                return 0
            else:
                self.ConditionalPTH[self.unpack(Test)] = HistoryCounter/HistoryDenumenatorCounter
                return HistoryCounter/HistoryDenumenatorCounter

    # gives the Matrix corresponding to the index of starting and ending columns of P_TaoH in the fat matrix (Fat Matrix has extended test size compared to usual P_TH)
    def indexMatrix(self, ao: list, Matrix):
        if((ao[0],ao[1]) in self.DenoisingPTH):
            return self.DenoisingPTH[(ao[0],ao[1])]
        List = []
        INDEX = []
        TEST=self.TESTelemet
        for element in TEST:
            List.append(ao + element)
        for element in List:
            INDEX.append(self.indexMap[self.unpack(element)])
        B_AO = np.matrix(Matrix[:, INDEX[0]])
        for index in INDEX[1:]:
            B_AO = np.hstack((B_AO, Matrix[:, index]))
        self.DenoisingPTH[(ao[0], ao[1])]=B_AO
        return B_AO

    # create all the permutation of 0,1,2 of given size i
    def combinationNaive(self):
        T1 = [[]]
        T2 = []
        for i1 in self.options:
            for i2 in self.observations:
                T1.append([i1, i2])
        return T1


    # create all the permutation of 0,1,2 of given size i
    def combination(self,Flag=False):
        T1 = [[]]
        T2 = []
        EqualSum=[]
        for i1 in self.options:
            L1 = [[]]
            for i2 in self.observations:
                L1.append([i1, i2])
                T1.append([i1, i2])
                for i3 in self.options:
                    L = []
                    L.append([i1, i2])
                    for i4 in self.observations:
                        T2.append([i1, i2, i3, i4])
                        L.append([i1, i2, i3, i4])
                    EqualSum.append(L)
            EqualSum.append(L1)
        if(Flag):
            List = T1 + T2
            for element in List:
                self.indexMap[self.unpack(element)] = List.index(element)
            for element in EqualSum:
                FinalEqualSum = []
                for l in element:
                    FinalEqualSum.append(self.indexMap[self.unpack(l)])
                self.FinalEqualSum.append(FinalEqualSum)
        return T1 + T2

    def combinationExtended(self):
        T1 = [[]]
        T2 = []
        T3=[]
        for i1 in self.options:
            for i2 in self.observations:
                T1.append([i1, i2])
                for i3 in self.options:
                    for i4 in self.observations:
                        T2.append([i1, i2, i3, i4])
                        for i5 in self.options:
                            for i6 in self.observations:
                                T3.append([i1, i2, i3, i4,i5,i6])
        List=  T1 + T2 +T3
        for element in List:
            self.indexMap[self.unpack(element)]=List.index(element)
        return T1 + T2+T3

    def TwoGivenListIsEqual(self, L1, L2):
        for i in range(L1.__len__()):
            if (L1[i] != L2[i]):
                return False
        return True

    # construct the list for second constraints in denoising algorithm
    def FindEqualElementsOfDynamicMatrix(self, Test, History):
        MainList = []  # MainList consist of all combination Test+History where in each of its element first entity is the combination, second entity is the index of corresponding element in dynamic matrix
        ListOfEqualEntities = []
        for test in Test:
            for history in History:
                MainList.append([history + test, [self.indexMap[self.unpack(history)], self.indexMap[self.unpack(test)]]])
                self.MapForListofEqualEntitis[self.unpack(history + test)]=[]
        print("Yes")
        print(len(MainList))
        for element in MainList:
            self.MapForListofEqualEntitis[self.unpack(element[0])].append(element[1])
        for keys in self.MapForListofEqualEntitis.keys():
            if self.MapForListofEqualEntitis[keys].__len__()>1:
                ListOfEqualEntities.append(self.MapForListofEqualEntitis[keys])
        return ListOfEqualEntities

#determines the next position of robot based on action
    def Move(self,position,action):
        Newposition=[]
        if(action==0):
            Newposition= [position[0],position[1]-1]
        elif(action==1):
            Newposition= [position[0], position[1] + 1]
        elif (action == 2):
            Newposition= [position[0]-1, position[1] ]
        elif (action == 3):
            Newposition= [position[0]+1, position[1] ]
        if (self.unpack(Newposition) in self.blocks):
            return [-1, -1]
        else:
            return Newposition

    def Update_Mapforcounting(self, Test):
        for data in Test:
            z=[]
            for k in data:
                z.append(k)
                if((self.unpack(z) in self.MapForCounting)==False):
                    self.MapForCounting[self.unpack(z)]=1
                else:
                    self.MapForCounting[self.unpack(z)]+=1

# return the option that has maximum value function
    def Max_option(self,ExtraTree,state):
        Max=ExtraTree.predict(np.hstack((state,np.matrix([[0]]))))
        option=0
        for i in range(1,len(self.options)):
            if(ExtraTree.predict(np.hstack((state,np.matrix([[i]]))))>Max):
                Max=ExtraTree.predict(np.hstack((state,np.matrix([[i]]))))
                option=i
        return option

#return the option of trajectory based on the possible valid action for that state
    def getAction(self):
        step=0
        r=random.uniform(0,1)
        for l in range(len(self.options)):
            if(step<=r<step+self.optionProbabilityHenkel[l]):
                return l
            else:
                step+=self.optionProbabilityHenkel[l]
        return len(self.optionProbabilityHenkel)-1


    def unpack_vector(self,v):
        A=[]
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                A.append(v[i,j])
        return np.array(A)
    def filter_New_PTH(self,Matrix):
        for i in range(Matrix.shape[0]):
            for j in range(Matrix.shape[1]):
                if(Matrix[i,j]<0):
                    Matrix[i, j]=0

    # Given a list with elements produces a string containing the elements of the list
    def unpack(self, list):
        s = ''
        j=0
        for l in list:
            if(j!=0 ):
                s = s +','+ str(l)
            else:
                s=s+str(l)
            j+=1
        return s



