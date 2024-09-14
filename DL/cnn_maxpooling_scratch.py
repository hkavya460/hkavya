import numpy as np
# CNNs are essential in many applications, from autonomous driving to medical image analysis.

#cretae the input matrix
def conv_layer(input_matrix):
    print("the size of input is ",input_matrix.shape)
    filter = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    print(filter.shape)
    h,w = input_matrix.shape
    output = np.zeros((h-2,w-2))
    for i in range(h-2):
        for j in range (w-2) :
            sub_matrix  =  input_matrix[i:(i+3),j:(j +3)]
            output[i,j]   = np.sum(sub_matrix * filter)
    return output


#pooling
#straided = 2  #filter 2*2
def max_pooling(output):
    h,w =output.shape
    new_h = h //2
    new_w = w //2
    max_pool = np.zeros((new_h,new_w))
    for i in range(new_h):
        for j in range(new_w):
            max_pool[i,j] = np.amax(output[2 *i : (2*i + 2) , 2*j : (2*j + 2)],axis=(0,1))
            # max_pool[i,j] = np.amax(output[i : (i +2) ,j: (j +2)],axis=(0,1))
    return max_pool

#fully_connected layer



def main():
    # input_matrix = np.array([[3,0,1,2,7,4],[1,5,8,9,3,1],[2,7,2,5,1,3],[0,1,3,1,7,8],[4,2,1,6,2,8],[2,4,5,2,3,9]])
    input_matrix = np.random.rand(32,32)
    output = conv_layer(input_matrix)
    print("the size of conv layer with 3 * 3 filter is  ",output.shape)

    max_pool =  max_pooling(output)
    print("the size of the max_pooling with 2*2 filter with striding 2 is ", max_pool.shape)

if __name__=="__main__":
    main()















