#in this class we are going to implement algorithm 2 of Melanie's Thesis and learn A=QR
# ***This class has been implemented via parallel programming for faster runtime***

import numpy as np
import copy
import math
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
class newAlgorithm:


    def __init__(self, rank,alpha, DynamicMatrix, weights, ListEqualEntitities,ListOfSumEqualEntities,lambda1, step,numberOfStep,FileName):
        # np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)
        self.rank=rank
        self.rows = DynamicMatrix.shape[0]
        self.columns=DynamicMatrix.shape[1]
        self.DynamicMatrix= DynamicMatrix
        self.ListEqualEntitities=ListEqualEntitities #List of Equal entities in Dynamic Matrix (This list contains sublists with elements in the format of [i,j] where i specify its row and j its column )
        self.ListOfSumEqualEntities=ListOfSumEqualEntities
        self.currentQ=np.zeros((self.rows,self.rank)) #initialize Q
        self.currentR=np.zeros((self.rank,self.columns))  #initilize R
        self.weights = weights #our weights corresponding to weighted sum of our new objective
        self.lambda1=lambda1 #our regularization factor
        self.setShape()
        self.step=step
        self.alpha=alpha
        self.numberOfStep=numberOfStep
        self.FileName=FileName
        self.ZeroQRows, self.ZeroRColumns = [],[]
        # self.ListEqualEntitities=self.ScanListEqualEntities()
        # self.ListOfSumEqualEntities=self.ScanListSumEqualEntities()
        self.RangeQelements,self.RangeRelements,self.Qmap,self.Rmap=self.NewRange() #self.Qmap & self.Rmap provides a map for the place where previous indices in Dynamic matrx mapped to new indeces in reduced Q and R
        self.ListofequalityforR={} #it is a dictionary for storing the corresonding list of equal value of Dynamic Matrix for each member in Q
        self.ListofequalityforQ={} #it is a dictionary for storing the corresonding list of equal value of Dynamic Matrix for each member in R
        self.Qfilter,self.Rfilter=self.Scan_QR() #it is used in first constraint as a hash map to not consider zero value weights
        for i in self.RangeQelements:
            self.ListofequalityforQ[i]=[]
        for i in self.RangeRelements:
            self.ListofequalityforR[i]=[]
        self.flag=False #it is a flag for showing if we can use self.Listofequalityfor, self.ListofequalityforQ

#  Compute the minimum of our objective via gradient decent (Optimized for maximum performance and accurate result)
    def GradientEngine(self):
        self.currentQ, singularValues, self.currentR = np.linalg.svd(self.DynamicMatrix, full_matrices=False)
        self.currentR = np.diag(singularValues) * self.currentR
        self.currentR = self.currentR[0:self.rank,:]
        self.currentQ = self.currentQ[:, :self.rank]
        j = 0
        for i in self.ZeroQRows:
            self.currentQ = np.delete(self.currentQ, (i - j), axis=0)
            j += 1
        j = 0
        for i in self.ZeroRColumns:
            self.currentR = np.delete(self.currentR, (i - j), axis=1)
            j += 1
        PreviousObjective=0
        INCREASE=False
        DECREASE=False
        i=0
        alpha=self.alpha
        while(True):
            #This if else statement is just for increasing speed, when we prevent update we can use previous calculation to save time and just manipulate the step
            numberOfChildProcess=3
            if(self.flag==True):
                # pool = Pool(processes=4)  # used for parallel programming
                # Q1 = pool.apply_async(self.FirstConstraintQForChild, args=(self.RangeQelements,0,))
                # Q2 = pool.apply_async(self.SecondConstraintQForChild, args=(self.RangeQelements,0,))
                # step=math.ceil(len(self.RangeRelements)/2)
                # R11 = pool.apply_async(self.FirstConstraintRForChild, args=(self.RangeRelements[0:step], 0,))
                # R12 = pool.apply_async(self.FirstConstraintRForChild, args=(self.RangeRelements[step:], step,))
                # R21 = pool.apply_async(self.SecondConstraintRForChild, args=(self.RangeRelements[0:step], 0,))
                # R22 = pool.apply_async(self.SecondConstraintRForChild, args=(self.RangeRelements[step:], step,))
                # pool.close()
                # pool.join()
                # R1 = np.hstack((R11.get(), R12.get()))
                # R2 = np.hstack((R21.get(), R22.get()))
                # Q = Q1.get() + self.lambda1 * Q2.get()
                # R = R1 + self.lambda1 * R2
                pool = Pool(processes=5)  # used for parallel programming
                Q1 = pool.apply_async(self.FirstConstraintQForChild, args=(self.RangeQelements,0,))
                Q2 = pool.apply_async(self.SecondConstraintQForChild, args=(self.RangeQelements,0,))
                R1 = pool.apply_async(self.FirstConstraintRForChild, args=(self.RangeRelements,0,))
                R2 = pool.apply_async(self.SecondConstraintRForChild, args=(self.RangeRelements,0,))
                R3= pool.apply_async(self.ThirdConstraintRForChild, args=(self.RangeRelements,0,))
                # self.ThirdConstraintRForChild(self.RangeRelements,0)
                pool.close()
                pool.join()
                Q = Q1.get() + self.lambda1 * Q2.get()
                R = R1.get() + self.lambda1 * (R2.get()+alpha*R3.get())
            else:
                Q1=self.FirstConstraintQForChild(self.RangeQelements,0)
                Q2=self.SecondConstraintQForChild(self.RangeQelements,0)
                R1 = self.FirstConstraintRForChild(self.RangeRelements, 0)
                R2 = self.SecondConstraintRForChild(self.RangeRelements, 0)
                R3=self.ThirdConstraintRForChild(self.RangeRelements, 0)
                Q = Q1 + self.lambda1*Q2
                R = R1 + self.lambda1*(R2+alpha*R3)
            self.currentQ=self.currentQ-self.step*(Q)
            self.currentR = self.currentR - self.step * (R)
            Gradient=np.hstack((Q.T,R))
            # Answer1=self.currentQ*self.currentR-self.DynamicMatrix
            # Answer2=np.multiply(self.weights, Answer1)
            # Objective='{:.50f}'.format(np.trace(Answer2*(Answer1.T))) #the objective of our optimization
            GradientPrint='{:.50f}'.format(np.linalg.norm(Gradient))
            NewObjective=np.linalg.norm(Gradient)
            self.flag=True
            if(i>=self.numberOfStep):  #for my slow computer tests!
                break
            Indent=10**(-10) #it is our index to check whether the gradient decsent is working or not
            if(i>1 and abs(NewObjective-PreviousObjective)/PreviousObjective<Indent): #increases step size if change in objective is not noticeble
                break
            if(i>1 and NewObjective-PreviousObjective<0):
                DECREASE=True
                INCREASE=False
            elif(i>1 and NewObjective-PreviousObjective==0):
                DECREASE=False
                INCREASE=False
            elif(i>1 and NewObjective-PreviousObjective>0):
                INCREASE=True
                DECREASE=False
            if(INCREASE==True):
                self.step=self.step/1.1
            PreviousObjective = NewObjective
            if(math.isnan(float(NewObjective))):
                with open(self.FileName, "a") as myfile:
                    myfile.write("step" + str(i) + ":" + str(GradientPrint) + "....... Experiment has repeated due to nan ......." + '\n')
                print("step", i, ":", GradientPrint, " ....... Experiment has repeated due to nan ....... ",'\n')
                return False,self.step
            if(i%50==0):
                with open(self.FileName, "a") as myfile:
                    myfile.write("step" + str(i) + ":" + str(GradientPrint) + " .... Increase: "+ str(INCREASE)+" .... Decrease: "+ str(DECREASE)+ '\n')
            print("step", i, ":", GradientPrint, " .... Increase: ", str(INCREASE)," .... Decrease: ", str(DECREASE), '\n')
            i+=1
        q=np.matrix(np.zeros(self.rank))
        for i in self.ZeroQRows:
            self.currentQ=np.vstack((self.currentQ[:i,:],q,self.currentQ[i:,:]))
        r = np.matrix(np.zeros(self.rank)).T
        for i in self.ZeroRColumns:
            self.currentR = np.hstack((self.currentR[:, :i], r, self.currentR[:, i:]))
        self.ListofequalityforQ.clear()
        self.ListofequalityforR.clear()
        self.Qfilter.clear()
        self.Rfilter.clear()
        self.Rmap.clear()
        self.Qmap.clear()
        return self.currentQ,self.currentR

# Break the work of computation of gradient matrix for Q to multiple process
    def FirstConstraintQ(self,numberOfChildProcess=1):
        pool = ThreadPool(processes=numberOfChildProcess)  # used for parallel programming
        index=math.floor(len(self.RangeQelements)/numberOfChildProcess)
        ProcessList={} #Contains processes
        step=0
        for i in range(numberOfChildProcess):
            if(i!=numberOfChildProcess-1):
                ProcessList[i]=pool.starmap(self.FirstConstraintQForChild, zip([self.RangeQelements[step:step+index]],[step]))
                step+=index
            else:
                ProcessList[i]=pool.starmap(self.FirstConstraintQForChild, zip([self.RangeQelements[step:]],[step]))
        pool.close()
        pool.join()
        Q=ProcessList[0][0]
        for i in range(1,numberOfChildProcess):
            Q=np.vstack((Q,ProcessList[i][0]))
        return Q


# compute the first part of gradient matrix for parameters of matrix Q(the first sum !)
    def FirstConstraintQForChild(self,RANGEi,index):
        newQ1 = np.matrix(np.zeros((len(RANGEi), self.QShape[1])))
        # Index=RANGEi[0]
        for i in range(len(RANGEi)):
            for j in range(self.QShape[1]):
                Sum=0
                for k in self.Qfilter[RANGEi[i]]:
                    Sum=Sum-2*self.currentR[j,k]*self.weights[RANGEi[i],self.RangeRelements[k]]*(self.DynamicMatrix[RANGEi[i],self.RangeRelements[k]]-self.currentQ[i+index,:]*self.currentR[:,k])
                newQ1[i,j]=Sum
        return newQ1

#  Break the work of computation of gradient matrix for R to multiple process
    def FirstConstraintR(self,numberOfChildProcess=1):
        pool = ThreadPool(processes=numberOfChildProcess)  # used for parallel programming
        index = math.floor(len(self.RangeRelements)/numberOfChildProcess)
        ProcessList = {}  # Contains processes
        step=0
        for i in range(numberOfChildProcess):
            if (i != numberOfChildProcess - 1):
                ProcessList[i]=pool.starmap(self.FirstConstraintRForChild, zip([self.RangeRelements[step:step+index]],[step]))
                step += index
            else:
                ProcessList[i]=pool.starmap(self.FirstConstraintRForChild, zip([self.RangeRelements[step:]],[step]))
        pool.close()
        pool.join()
        R=ProcessList[0][0]
        for i in range(1,numberOfChildProcess):
            R=np.hstack((R,ProcessList[i][0]))
        return R

# compute the first part of gradient matrix for parameters of matrix R(the first sum !)
    def FirstConstraintRForChild(self, RANGEi,index):
        newR1 = np.matrix(np.zeros((self.RShape[0],len(RANGEi))))
        # Index = RANGEi[0]
        for i in range(self.RShape[0]):
            for j in range(newR1.shape[1]):
                Sum = 0
                for k in self.Rfilter[RANGEi[j]]:
                    Sum = Sum - 2 * self.currentQ[k, i] * self.weights[self.RangeQelements[k], RANGEi[j]] * (self.DynamicMatrix[self.RangeQelements[k], RANGEi[j]] - self.currentQ[k, :] * self.currentR[:, j+index])
                newR1[i, j] = Sum
        return newR1

# Break the work of computation of gradient matrix for Q to multiple process
    def SecondConstraintQ(self,numberOfChildProcess=1):
        pool = ThreadPool(processes=numberOfChildProcess)  # used for parallel programming
        index=math.floor(len(self.RangeQelements)/numberOfChildProcess)
        step=0
        ProcessList={} #Contains processes
        for i in range(numberOfChildProcess):
            if(i!=numberOfChildProcess-1):
                ProcessList[i]=pool.starmap(self.SecondConstraintQForChild, zip([self.RangeQelements[step:step+index]],[step]))
                step += index
            else:
                ProcessList[i]=pool.starmap(self.SecondConstraintQForChild, zip([self.RangeQelements[step:]],[step]))
        pool.close()
        pool.join()
        Q=ProcessList[0][0]
        for i in range(1,numberOfChildProcess):
            Q=np.vstack((Q,ProcessList[i][0]))
        return Q

# compute the second part of gradient matrix for parameters of matrix Q(the second sum for the first constraints!)
    def SecondConstraintQForChild(self,RANGEi,index):
        newQ2=np.matrix(np.zeros((len(RANGEi),self.QShape[1])))
        # Index = RANGEi[0]
        for i in range(len(RANGEi)):
            if (self.flag == False):
                for list in self.ListEqualEntitities:
                    boolean, newList, s = self.checkIfinListQ(list, RANGEi[i])
                    if (boolean):
                        for j in range(self.QShape[1]):
                            Sum = 0
                            for k in newList:
                                Sum = Sum + 2 * self.currentR[j, self.Rmap[s]] * (self.currentQ[i + index, :] * self.currentR[:, self.Rmap[s]] - self.currentQ[self.Qmap[k[0]],:] * self.currentR[:,self.Rmap[k[1]]])
                            newQ2[i, j] += Sum
            else:
                for list in self.ListofequalityforQ[RANGEi[i]]:
                    boolean, newList, s = True,list[0],list[1]
                    if (boolean):
                        for j in range(self.QShape[1]):
                            Sum = 0
                            for k in newList:
                                Sum = Sum + 2 * self.currentR[j, self.Rmap[s]] * (self.currentQ[i + index, :] * self.currentR[:, self.Rmap[s]] - self.currentQ[self.Qmap[k[0]],:] * self.currentR[:,self.Rmap[k[1]]])
                            newQ2[i, j] += Sum
        return newQ2

#  Break the work of computation of gradient matrix for R to multiple process
    def SecondConstraintR(self,numberOfChildProcess=1):
        pool = ThreadPool(processes=numberOfChildProcess)  # used for parallel programming
        index = math.floor(len(self.RangeRelements) / numberOfChildProcess)
        ProcessList = {} # Contains processes
        step=0
        for i in range(numberOfChildProcess):
            if (i != numberOfChildProcess - 1):
                ProcessList[i]=pool.starmap(self.SecondConstraintRForChild,zip([self.RangeRelements[step:step+index]],[step]))
                step += index
            else:
                ProcessList[i]=pool.starmap(self.SecondConstraintRForChild, zip([self.RangeRelements[step:]],[step]))
        pool.close()
        pool.join()
        R=ProcessList[0][0]
        for i in range(1,numberOfChildProcess):
            R=np.hstack((R,ProcessList[i][0]))
        return R

# compute the second part of gradient matrix for parameters of matrix R(the second sum for the first constraints!)
    def SecondConstraintRForChild(self,RANGEj,index):
        newR2=np.matrix(np.zeros((self.RShape[0],len(RANGEj))))
        # Index = RANGEj[0]
        for j in range(len(RANGEj)):
            if (self.flag == False):
                for list in self.ListEqualEntitities:
                    boolean, newList, s = self.checkIfinListR(list, RANGEj[j])
                    if (boolean):
                        for i in range(self.RShape[0]):
                            Sum = 0
                            for k in newList:
                                Sum = Sum + 2 * self.currentQ[self.Qmap[s], i] * (self.currentQ[self.Qmap[s], :] * self.currentR[:, j + index] - self.currentQ[self.Qmap[k[0]],:] * self.currentR[:,self.Rmap[k[1]]])
                            newR2[i, j] += Sum
            else:
                for list in self.ListofequalityforR[RANGEj[j]]:
                    boolean, newList, s = True,list[0],list[1]
                    if (boolean):
                        for i in range(self.RShape[0]):
                            Sum = 0
                            for k in newList:
                                Sum = Sum + 2 * self.currentQ[self.Qmap[s], i] * (self.currentQ[self.Qmap[s], :] * self.currentR[:, j + index] - self.currentQ[self.Qmap[k[0]],:] * self.currentR[:,self.Rmap[k[1]]])
                            newR2[i, j] += Sum
        return newR2

# compute the second part of gradient matrix for parameters of matrix R(the second sum for the first constraints!)
    def ThirdConstraintRForChild(self,RANGEj,index):
        newR2=np.matrix(np.zeros((self.RShape[0],len(RANGEj))))
        # Index = RANGEj[0]
        for element in self.ListOfSumEqualEntities:
            for j in range(newR2.shape[0]):
                i = 0
                Sum=0
                for entity in element:
                    if(i==0):
                        Sum-=self.currentR[j,self.Rmap[entity]]
                    else:
                        Sum+=self.currentR[j,self.Rmap[entity]]
                    i+=1
                i=0
                for entity in element:
                    if(i==0):
                        newR2[j,self.Rmap[entity]]-=2*Sum
                    else:
                        newR2[j, self.Rmap[entity]] += 2 * Sum
                    i+=1
        return newR2

#Assign the shape of Matrices R and Q
    def setShape(self):
        self.QShape=[self.rows,self.rank]
        self.RShape=[self.rank,self.columns]


# checks whether index i of Q entry exists in a list of equal entries
    def checkIfinListQ(self,list,i):
        q=copy.deepcopy(list)
        for element in q:
            if(element[0]==i):
                q.remove(element)
                self.ListofequalityforQ[i].append([q,element[1]])
                return True,q,element[1]
        return False,list,False



# checks whether index i of R entry exists in a list of equal entries
    def checkIfinListR(self,list,i):
        q=copy.deepcopy(list)
        for element in q:
            if(element[1]==i):
                q.remove(element)
                self.ListofequalityforR[i].append([ q, element[0]])
                return True,q,element[0]
        return False,list,False

#Scan Dynamic Matrix elements to see if a column or row of it is zero then report such column or row.
# (For reduction in the redundant Computation)
    def Scan(self):
        ZeroRColumns=[]
        ZeroQRows=[]
        # A=np.matrix(np.zeros(self.DynamicMatrix.shape[1]))
        # B=np.matrix(np.zeros(self.DynamicMatrix.shape[0]))
        # for i in range(self.DynamicMatrix.shape[0]):
        #     if(np.array_equal(self.DynamicMatrix[i],A)):
        #         ZeroQRows.append(i)
        # for j in range(self.DynamicMatrix.shape[1]):
        #     if (np.array_equal(self.DynamicMatrix[:,j].T, B)):
        #         ZeroRColumns.append(j)
        return ZeroQRows,ZeroRColumns

# delete the irrelevant entities in list of equal entities based on Scan function which detected useless elements
# to study in Dynamic Matrix
    def ScanListEqualEntities(self):
        NewList=[]
        for elements in self.ListEqualEntitities:
            NewElement = []
            for entity in elements:
                if (entity[0] in self.ZeroQRows or entity[1] in self.ZeroRColumns)==False:
                    NewElement.append(entity)
            if(len(NewElement)>1):
                NewList.append(NewElement)
        return NewList

# delete the irrelevant entities in list of equal entities based on Scan function which detected useless elements
# to study in Dynamic Matrix
    def ScanListSumEqualEntities(self):
        NewList = []
        for elements in self.ListOfSumEqualEntities:
            NewElement = []
            i=0
            for entity in elements:
                if ((entity in self.ZeroRColumns) and (i==0)) == True:
                    break
                if (entity in self.ZeroRColumns) == False:
                    NewElement.append(entity)
                i+=1
            if (len(NewElement) > 1):
                NewList.append(NewElement)
        return NewList

# This is a function for initiallizing the hashmap of Qfiter and Rfilter
    def Scan_QR(self):
        Q={}
        R={}
        for q in self.RangeQelements:
            Q[q]=[]
            for r in range(len(self.RangeRelements)):
                if(self.weights[q,self.RangeRelements[r]]!=0):
                    Q[q].append(r)
        for r in self.RangeRelements:
            R[r] = []
            for q in range(len(self.RangeQelements)):
                if (self.weights[self.RangeQelements[q], r] != 0):
                    R[r].append(q)
        return Q,R




#Based on the ZeroQRows & ZeroRColumns creates the range for column of matrix R and rows of matrix Q that we are
#going to calculate their modeified entities based on our denoising algorithm
    def NewRange(self):
        QList=[]
        Qmap = {}
        ZeroQ=copy.deepcopy(self.ZeroQRows)
        j=0
        for i in range(self.DynamicMatrix.shape[0]):
            if(len(ZeroQ)==0):
                QList.append(i)
                Qmap[i] = j
                j += 1
                continue
            elif(i!=ZeroQ[0]):
                QList.append(i)
                Qmap[i] =j
                j += 1
            else:
                del ZeroQ[0]
        RList = []
        ZeroR = copy.deepcopy(self.ZeroRColumns)
        j = 0
        Rmap = {}
        for i in range(self.DynamicMatrix.shape[1]):
            if (len(ZeroR) == 0):
                RList.append(i)
                Rmap[i] = j
                j += 1
                continue
            elif (i != ZeroR[0]):
                RList.append(i)
                Rmap[i] = j
                j += 1
            else:
                del ZeroR[0]
        return QList,RList,Qmap,Rmap




















