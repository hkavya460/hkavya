import numpy as np

# Create the input matrix
input_matrix = np.array([
    [3,0,1,2,7,4],
    [1,5,8,9,3,1],
    [2,7,2,5,1,3],
    [0,1,3,1,7,8],
    [4,2,1,6,2,8],
    [2,4,5,2,3,9]
])

# Create the filter (3x3)
filter = np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
])

# Dimensions of the input matrix
h, w = input_matrix.shape

# Output matrix after convolution
output = np.zeros((h-2, w-2))

# Perform convolution operation
for i in range(h-2):
    for j in range(w-2):
        # Extract the submatrix
        submatrix = input_matrix[i:i+3, j:j+3]
        # Perform element-wise multiplication and sum the result
        output[i, j] = np.sum(submatrix * filter)

print("Convolution Output:")
print(output)
