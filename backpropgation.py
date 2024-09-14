import numpy as np
np.random.seed(42)


def two_layer_backpropation():
    num_neurons = 1
    num_layers = 1
    n_inputs =4
    y = 1
    weights = np.random.rand(num_layers, n_inputs).round(2)
    bias = np.random.rand(num_neurons).round(2)
    x = np.array([[1], [0], [1], [0]])
    h1 = np.dot(weights,x) + bias
    relu = np.maximum(0,h1)
    # print(relu)
    y_p  = relu
    #loss function
    l = 1/2 * ((y - y_p) **2 )
    print(f"loss is ",l)

    #backpropagtion
    dl_dy_p  = - (y - y_p)
    # derivative of h1
    dl_dhidden  = (h1 > 0 ).astype(float)
    dl_dh1 = dl_dy_p * dl_dhidden
    # derivative wrt to x
    dl_dw = dl_dh1 * x.T
    new_weight = weights - 0.01 * dl_dh1
    print("new weights are", new_weight)


#gradients are
    print("Gradient w.r.t activation (dl_da):", dl_dy_p)
    print("Gradient w.r.t pre-activation (dl_dz):", dl_dh1)
    print("Gradients w.r.t weights (dl_dw):", dl_dw)


# 3 layer backpropagtion
def three_layer_backpropgation():

    num_inputs = 4
    inputs = np.random.rand(4,1)
    num_hidden_lays = 3
    num_neurons_1 = 3
    num_neurons_2 = 2
    num_neuron_3 = 1
    y = 2
    w_1 = np.random.rand(num_neurons_1,num_inputs)

    w_2 = np.random.rand(num_neurons_2,num_neurons_1)
    w_3 = np.random.rand(num_neuron_3,num_neurons_2)
    b_1 = np.random.rand(num_neurons_1,1)
    b_2 = np.random.rand(num_neurons_2,1)
    b_3 = np.random.rand(num_neuron_3,1)
    #feed foreward
    h1 = np.dot(w_1,inputs) + b_1

    act_1 = np.maximum(0,h1)
    # print(act_1.shape)

    h2 = np.dot(w_2,act_1) + b_2
    act_2 = np.maximum(0,h2)

    h3 = np.dot(w_3,act_2) + b_3
    act_3 = np.maximum(0,h3)
    y_p = act_3

    l = 1/2 * ((y - y_p)**2)

    print("loss in three layer is ", l)

    dl_dact3 = -1* (y - act_3)
    # print(dl_dact3)
    dl_dh3 = (h3 > 0).astype(float)
    # print(dl_dh3)
    dl_dh_3 = dl_dact3 * dl_dh3
    dh3_da_2 =  (act_3 > 0  ).astype(float)



def main():
    two_layer_backpropation()
    three_layer_backpropgation()

main()
