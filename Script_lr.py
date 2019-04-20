# -*- coding: utf-8 -*-
"""
Created on Thu Feb 07 17:45:12 2019

@author: Sam
"""


#@author: Sam

import scipy as sc;
import numpy as np;
import os
import matplotlib as mp
os.system('cls')
# Input training and test data
#train= np.loadtxt("F:\UTA\EE5359_ML\Assignment 3\Training_edit.txt", dtype='f', delimiter=',');
#test=np.loadtxt("F:\UTA\EE5359_ML\Assignment 3\Test_edit.txt", dtype='f', delimiter=',');

#################### Pre Processing the data file format ##############################
with open("F:\UTA\EE5359_ML\Assignment 3\Training.txt") as f:
    array = []
    for line in f:
        a=line.replace('   ',',' )
        a=a.replace('  ',',')
        a=a.replace (' ','  ')
        a=a.replace('  ', ',')
        a=a.replace('\n',"")
        a=a.lstrip(",")
        a=a.rstrip(",")
        a=a.split(',')
        array.append(a)
   # content = f.readlines()
   # b=[int(a) for i in a]
    b= np.asarray(array)
    train=b.astype(int)
    
    
with open("F:\UTA\EE5359_ML\Assignment 3\Test.txt") as f2:
    array2 = []
    for line in f2:
        aa=line.replace('   ',',' )
        aa=aa.replace('  ',',')
        aa=aa.replace (' ','  ')
        aa=aa.replace('  ', ',')
        aa=aa.replace('\n',"")
        aa=aa.lstrip(",")
        aa=aa.rstrip(",")
        aa=aa.split(',')
        array2.append(aa)
   # content = f.readlines()
   # b=[int(a) for i in a]
    bb= np.asarray(array2)
    test=bb.astype(int) 


######################### Pre Processing the data file format end ######################################


############## SETTINGS TO DEFINE THE ORDER OF POLYNOMIAL AS A USER INPUT OR NOT##################
#train=train[1:200,:]
N_c,p_c=train.shape; # dimension of train data - dummy varaibales
N_t,p_t=test.shape; # dimension of test data - dummy varaibales
print p_c;
print N_c;


#order=(raw_input("Enter the max polynomial order needed: "))
#o=int(order)
op=[1,2] # Order of polynomials 1 and 2

color=iter(mp.pyplot.cm.rainbow(np.linspace(0,1,4)))
mp.pyplot.close('all')

mp.pyplot.figure(1)
mp.pyplot.grid()

#Looping in  polynomials 1 and 2
for l in range(0,2):
    o=op[l]
############## SETTINGS TO DEFINE THE ORDER OF POLYNOMIAL AS A USER INPUT OR NOT - END ###############
   
    
    
    
#######################TEST FILE PRE_PROCESSING START #########################3
    #Appending and pre-processing test file
    y_test=test[:,p_t-1:p_t].flatten()
    xapp_test=np.ones(N_t)
    x_test=np.vstack((xapp_test,np.transpose(test[:,0:p_c-1])))
    for j in range(2,o+1):
            x_test=np.vstack((x_test,np.power(x_test[1:p_c+1,:],j))); # Test data working variables
    
    #Mapping class other than 0 as class 1 for testing data
    for j in range(0,N_t):
            if y_test[j]>1:
                y_test[j]=1
            else:
                y_test[j]=0
###########################TEST FILE PRE_PROCESSING END ##########################
    
    
    
    #Percentage of training data to be used
    train_per=[.25,.5,.75,1]
    #Variables to store misclassification data for plotting
    plotx=[]
    plotytr=[]
    plotyts=[]
    #Looping between different values of training data
    
    
    for i in range(0,4):
        ############################# PRE_PROCESSING TRAINING FILE ###############################
        num_selections = int(np.ceil(train_per[i]*N_c))
        plotx.append(num_selections)
        #idx = np.random.randint(N_c,size=int(np.ceil(train_per[i]*N_c)))
        #print idx.shape
        new_list = train[0:num_selections,:]
    
        train_ed=np.asarray(new_list)
        # train_ed=train
        x=np.transpose(train_ed[:,0:p_c-1]);#Train data real variable
        y=train_ed[:,(p_c)-1:p_c].flatten();#Train data real variable
        #print "Value of x for dataset with percentage "+str(train_per[i]*100)+"\n"
        #print x
        p,N=x.shape; #Shape of training data in real working variables
    
        xapp=np.ones(N) #N number of train data
        x=np.vstack((xapp,x))
        #print x
        B=np.array(np.zeros((p*(o)+1)))
        for j in range(2,o+1):
            x=np.vstack((x,np.power(x[1:p+1,:],j)));
            
        #Mapping class other than 0 as class 1 for testing data
        for j in range(0,N):
            if y[j]>1:
                y[j]=1
            else:
                y[j]=0
        ############################# PRE_PROCESSING TRAINING FILE END ###############################
     
        ########################## LOGISTIC REGRESSION WITH NEWTON RAPHSON ############################
        print "Misclassification for training dataset with "+str(int(train_per[i]*100))+" % of training data with "+ str(o)+" degree"
        
        for k in range(0,20):
            c1=np.matmul(B,x)
            #c2=np.log(1+np.exp(c1));
            pr0=(np.exp(c1))/(1+np.exp(c1))
            pr_mat=np.asmatrix(pr0)
            pr1=(1-pr0)
            pr1_mat=np.asmatrix(pr1)
            db=np.matmul(x,(y-pr0))
            #ddb=np.matmul(np.transpose(np.transpose(np.asmatrix(x))),(np.matmul(np.transpose(pr1_mat),pr_mat)))
            #W=np.matmul(ddb,np.transpose(np.asmatrix(x)))
            Bx=B;
            W=(np.matmul(np.transpose(np.diag(pr1)),np.diag(pr0)))
            X=np.transpose(np.asmatrix(x))
            ddb=np.matmul(np.transpose(X),np.matmul(W,X))
            B_up=np.matmul(np.linalg.inv(ddb),np.transpose(np.asmatrix(db)))
            B=B+np.transpose(B_up)
            B=(np.asarray(B)).flatten();
            if (np.linalg.norm(Bx-B)) <.001:
                print "Breaking the loop at "+str(k) + " iteration"
                break;
            #print "Value of B for dataset with percentage "+str(train_per[i]*100)+"\n"
            #print B
        
        C=np.matmul(B,x)
        Pr= pr=(np.exp(C))/(1+np.exp(C))
        P_est=(np.rint(Pr))
        
        C_test=np.matmul(B,x_test)
        Pr_test= pr=(np.exp(C_test))/(1+np.exp(C_test))
        P_est_test=(np.rint(Pr_test))
        
      
    
        E_tr=float(abs(sum(P_est-y))/N)
        print "Number of misclassification for  "+str(int(train_per[i]*100))+" % training data is: " +str(float(E_tr))
        plotytr.append(E_tr)
        E_tst=float(abs(sum(P_est_test-y_test))/N_t)
        plotyts.append(E_tst)
        print "Number of misclassification for  "+str(int(train_per[i]*100))+" % in testing data is: " +str(float(E_tst))
        print "\n"
    
      ########################## LOGISTIC REGRESSION WITH NEWTON RAPHSON ############################
     
    
    ########################### PLOT FIGURES #####################################################3
    c=next(color)
    d=next(color)
    mp.pyplot.plot(plotx,plotyts,'-b', c=c, label='Test misclassification '+str(o)+' degree')
    mp.pyplot.plot(plotx,plotytr, '-r', c=d, label='Train misclassification'+str(o)+' degree')
    mp.pyplot.legend();
    ########################### PLOT FIGURES #####################################################3