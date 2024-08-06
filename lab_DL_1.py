import numpy as np
import matplotlib.pyplot as plt
import math
# values between -10 and 10. Call this list as z. Implement the following functions and its derivative. Use class notes to find the expression for these functions. Use z as input and
# plot both the function outputs and its derivative outputs.
# a. Sigmoid
# b. Tanh
# c. ReLU (Rectified Linear Unit)
# d. Leaky ReLU
# e. Softmax
# # Implement the following functions and its derivative
#sigmoid function  = 1 / 1+ e^ -x used in logistic regression for binary classification problems
#2. Write down the observations from the plot for all the above functions in the code.
# a. What are the min and max values for the functions
# b. Are the output of the function zero-centred
# c. What happens to the gradient when the input values are too small or too big

def sigmoid_function(z):
    g = 1 / (1 + np.exp(-z))
    #derivative  of the sigmoid  function
    g_1 = 1 / (1+ (np.exp(-z))) * (1 - 1 / (1+ (np.exp(-z))))
    plt.plot(z,g,label='sigmoid function',color='red')
    plt.plot(z, g_1,label='Derivative of sigmoid function',color='blue')
    plt.axvline(0,color='black',linestyle="--")
    plt.xlabel('z values')
    plt.ylabel('sigmoid function')
    plt.title('plot of sigmoid and derivative of sigmoid function')
    plt.legend()

    plt.show()
    print("the minimum value of sigmoid function is ",min(g).round(3))
    print("the maximum value of sigmoid function is ", max(g).round(3))
    return g_1
    #tanh function or hyperbolic tangent function

def tanh_function(z):
    f = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    plt.plot(z,f,label="Tanh function")
    plt.xlabel('z values')
    # #derivative of the tanh_function
    f_1 = 1 - ((np.exp(z) - np.exp(-z)) **2) / ((np.exp(z) + np.exp(-z))**2)
    plt.plot(z,f_1,label="derivative of tanh")
    plt.ylabel("Tanh functions")
    plt.axvline(0 ,color='black',linestyle='--')
    plt.legend()
    plt.show()
    print("the minimum value of tanh function is ", min(f).round(3))
    print("the maximum value of tanh function is ", max(f).round(3))
    return f_1,f

def relu_function(z):
    f = np.maximum(0,z)

    # plt.show()
    #derivative of the relu function
    f_2 =[]
    for i in z:
        if i <= 0:
            i =0
            f_2.append(i)
        else:
            i = 1
            f_2.append(i)
    plt.plot(z, f,label='Relu function')
    plt.xlabel('z values')
    plt.ylabel('relu function')
    plt.show()
    plt.plot(z, f_2,color='cyan',label='Derivative of the relu function')
    plt.ylabel('Derivative of relu functions')
    plt.show()
    print("the minimum value of relu function is ", min(f).round(3))
    print("the maximum value of relu function is ", max(f).round(3))
    return f

def leaky_relu(z):
    f = np.maximum(0.01*z , z)
    plt.plot(z, f)
    plt.xlabel('z values')
    plt.ylabel('Leaky_relu function')
    plt.show()
    print("the minimum value of leaky relu function is ", min(f).round(3))
    print("the maximum value of leaky relu function is ", max(f).round(3))
    return f

def softmax_function(z):
    f = np.exp(z) / (sum(np.exp(z)))
    #derivative of the softmax function
    # z(i) * 1-z(j)  if i=j
    # - z(i) * z(j) if i != j
    plt.plot(z,f)
    plt.xlabel('Z values')
    plt.ylabel('Softmax function')
    plt.show()

##############################################

def main():
    z = np.linspace(-10, 10, num=100)
    g_1 = sigmoid_function(z)
    f_1,f = tanh_function(z)
    f = relu_function(z)
    f = leaky_relu(z)
    softmax_function(z)

if __name__=="__main__":
    main()


