import numpy as np
import matplotlib.pyplot as plt

def linear_coef(x,y) :
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
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([1,3,2,5,7,8,8,9,10,12])

    theta0,theta1 = linear_coef(x,y) 
   
    print('Hypothetical line is:: y = ',theta0,' + ',theta1,'*x',end=' ')
    plt.scatter(x,y,color="r")
    plt.plot(x,theta0+x*theta1,color="b")
    plt.title("Linear regression using square fit line")
    plt.xlabel('x',color='g')
    plt.ylabel('y/H(x)',color='g')
    plt.show()
main()

