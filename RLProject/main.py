
# ::::::Key::::: Move-Forward:0, Slight-Right-Turn:1, Sharp-Right-Turn:2, Slight-Left-Turn:3


import random
import time
import numpy as np
from ConstructEnvironment import *

def main():
    start_time = time.time()
    print("Started...")
    Result2=[]
    Result1=[]
    File2 = "result.txt"  # for storing results
    deleteContent(File2)
    File = "log.txt"  # for storing detail information
    deleteContent(File)
    with open(File, "a") as myfile:
        myfile.write("started .... " + '\n')
    with open(File2, "a") as myfile2:
        myfile2.write("started .... " + '\n')
    r1 = []
    r2=[]
    r11=[]
    r22=[]
    for step in [0.001, 0.005,0.0001]:
        for lamda1 in [10, 1,0.1 ]:
            for alpha in [1,0.1,0.01,0]:
                start_time = time.time()
                # r1=[]
                # r2=[]
                # r11=[]
                # r22=[]
                Ratio=[]
                for i in [500,700,900,1100,1300]:
                    L1 = []
                    L2 = []
                    L11=[]
                    L22=[]
                    Sparsity=[]
                    for j in range(10):
                        A = ConstructEnvironment( gamma=0.9,alpha=alpha,stepSize=step, rank=10,
                                                 lambda1=lamda1, numIteration=100, FileName=File)
                        L1.append(A.R1)
                        print(A.R1)
                        # L11.append(A.R1_train)
                        # L22.append(A.R2_train)
                        Sparsity.append([A.Ratio_Zero,A.NumberOfZero])
                    print("Sum Results: ", sum(L1)/float(len(L1)))
                    r1.append(L1)
                    r2.append(L2)
                    r11.append(L11)
                    r22.append(L22)
                    Ratio.append(Sparsity)
                with open(File2, "a") as myfile2:
                    myfile2.write('Results For step size equal to: ' + str(step) + ' and lamda1: ' + str(lamda1) + '\n')
                    myfile2.write('Ratio Zero in Henkel Matrix: ' +str(Ratio)+'\n')
                    myfile2.write("Test Error Algorithm without Denoising On Test Data: " + str(r1) + '\n')
                    myfile2.write("Test Error Algorithm with Denoising On Test Data: " + str(r2) + '\n')
                    myfile2.write("Train Error Algorithm without Denoising On Test Data: " + str(r11) + '\n')
                    myfile2.write("Train Error Algorithm with Denoising On Test Data: " + str(r22) + '\n')
                    myfile2.write("Running time: " + str(time.time() - start_time))
                print('Results For step size equal to: ', str(step), ' and lamda1: ', str(lamda1), '\n')
                print("Error Algorithm without Denoising On Test Data: ", Result1)
                print("Error Algorithm with Denoising On Test Data: ", Result2)






# Makes the test set trajectories bigger
def Modify(list):
    L=[]
    for i in range(0,len(list)-1,2):
        L.append(list[i]+list[i+1])
    return L


def deleteContent(fName):
    with open(fName, "w"):
        pass
if __name__ == '__main__':
    main()

