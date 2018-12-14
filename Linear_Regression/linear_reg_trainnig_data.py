#linear regression using square fit error method trainning from train.csv file
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

def linear_coef(x,y):
    n = np.size(x)
    x_mean,y_mean = np.mean(x),np.mean(y)
    s_xy = 0;
    s_xx = 0;
    for i in range (0,n):
        s_xy = s_xy+x[i]*y[i] 
    s_xy = s_xy - n*x_mean*y_mean

    for i in range (0,n):
        s_xx = s_xx+x[i]*x[i] 
    s_xx = s_xx - n*x_mean*x_mean
    theta1 = s_xy/s_xx
    theta0 = y_mean - theta1*x_mean
    return theta0,theta1


def main():
    
    filename = 'train.csv'
    data = pd.read_csv(filename) 
    data=np.array(data)
    x = data[:,[0]]
    y = data[:,[1]]
    
    theta0,theta1 = linear_coef(x,y)
    print(theta0,theta1 )
    plt.scatter(x,y,color="r")
    plt.plot(x,theta0+x*theta1,color="b")
    plt.title("Linear regression - Square fit method")
    plt.xlabel('x',color='g')
    plt.ylabel('y/H(x)',color='g')

    #test data 
    test = pd.read_csv('test.csv')
    test = np.array(test)
    x = test[:,[0]]
    y = test[:,[1]]
    plt.scatter(x,theta0+x*theta1,color="g")
    plt.show()
main()


