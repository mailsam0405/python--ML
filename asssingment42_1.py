#cal mean of x 
# mean of y 
# slop of m 
# intercept c

import numpy as np
import matplotlib .pyplot as plt
import pandas as pd

def marvellousregration():
    #data set load
    X=np.array([1,2,3,4,5])
    Y=np.array([3,4,2,4,5])
    print ("indipendent variable:",X)
    print ("dipendent variable:",Y)
    

    #calculate mean of x & y
    M_X=np.mean(X)
    M_Y=np.mean(Y)

    print("Mean of X is :",M_X)
    print("Mean of Y is :",M_Y)
    

    #calculate m
    #formula is x-mean(x)*y-mean(y)/x-mean(x)**2


    m=np.sum((X-M_X)*(Y-M_Y))/ np.sum((X-M_X)**2)
    print("Slop of m is :",m)
    
    #calculate intercept c
    #formula c=mean(y)-m*mean(x)

    c=M_Y-m*M_X
    print("intercept of c is :",c)

    #calculate regration equation 
    #formula y=m(x)+c
             #y=0.4*6+2.4

    Result=Y=m*6+c
    print("Predicted Y for X=6:",Result) 




def main():
    marvellousregration()

if __name__=="__main__":
    main()    