#linear regression using gradient descent method trainning from train.csv file
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

def summation_0(theta0,theta1,x,y):
    n = np.size(x)
    sum = 0
    for i in range (0,n):
        sum = sum+(theta0 + theta1*x[i] - y[i])
    return sum

def summation_1(theta0,theta1,x,y):
    n = np.size(x)
    sum = 0
    for i in range (0,n):
        sum = sum+(theta0 + theta1*x[i] - y[i])*x[i]
    return sum

#here alpha is the learning rate
def cost_minimisation(x,y,alpha):
    n = np.size(x)
    theta0,theta1 = 2,3
    for i in range (1000):
        temp0 = theta0 - alpha*summation_0(theta0,theta1,x,y)/n
        temp1 = theta1 - alpha*summation_1(theta0,theta1,x,y)/n
        theta0 = temp0
        theta1 = temp1
    return theta0,theta1


def main():
    
    filename = 'train.csv'
    data = pd.read_csv(filename) 
    data=np.array(data)
    x = data[:,[0]]
    y = data[:,[1]]
    alpha = float(input("Enter the learning rate::"))
    theta0,theta1 = cost_minimisation(x,y,alpha)
    print(theta0,theta1 )
    plt.scatter(x,y,color="r")
    plt.plot(x,theta0+x*theta1,color="b")
    plt.title("Linear regression -Gradient Descent ")
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


