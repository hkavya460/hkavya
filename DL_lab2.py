import  numpy as np
np.random.seed(42)
def relu_function(x):
    return np.maximum(0,x)
#implementing the feed foreward network using the relu function
def two_layer_ffn():
    num_neurons = 4
    num_layers = 1
    input_values = np.array([[1],[0],[1],[0]])
    w = np.array([20,-20,20,20]).reshape(1,4)
    b = np.array([-10])
    # calculating the
    z = np.dot(w,input_values) + b
    #activation function calling the relu function
    act = np.maximum(0,z)
    print("the activation function of one hidden layer is ",act)

# Implement forward pass for the above two networks. Print activation values for each
# neuron at each layer.
def three_layer_ffn():
    # without the loop
    hidden_layers = 3
    num_neurons = 4
    h_1 = 3
    h_2 = 2
    h_3 = 1
    #based on number of hidden layer it should creating the weights
    b1 = np.random.rand(h_1,1)
    b2 = np.random.rand(h_2,1)
    b3 = np.random.rand(h_3,1)
    input_values = np.array([[1],[0],[1],[0]])
    #first layer calculation
    w_1 = np.random.rand(h_1, num_neurons)
    z_1 = np.dot(w_1, input_values) + b1
    act_1 = relu_function(z_1)
    # 2nd layer
    w_2 = np.random.rand(h_2, h_1)
    z_2 = np.dot(w_2, act_1) + b2
    act_2 = relu_function(z_2)
    # 3rd layer
    w_3 = np.random.rand(h_3, h_2)
    z_3 = np.dot(w_3, act_2) + b3
    act_3 = relu_function(z_3)
    # Print activation values for each layer
    print("Activation values of the first hidden layer:",act_1)
    print("Activation values of the second hidden layer:",act_2)
    print("Activation values of the third hidden layer:",act_3)
    #using the loop
    for i in range(hidden_layers):
        z = np.dot(w_1[0],input_values)+ b1
        act_1 = relu_function(z)
        #move to the second layer
        z_2 = np.dot(w_2[1], act_1) + b2
        act_2 = relu_function(z_2)
        #move to the 3rd layer
        z_3 = np.dot(w_3[-1], act_2) + b3
        act_3 = relu_function(z_3)

    # print("the activation value of the first hidden layer ",act_1.max())
    # print("the activation value of the second  hidden layer ", act_2.max())
    # print("the activation value of the third  hidden layer ", act_3.max())

def  main():
    x = np.linspace(-10,10,100)
    two_layer_ffn()
    relu_function(x)
    three_layer_ffn()

main()



