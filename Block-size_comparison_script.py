
# %%
# Python
import pandas as pd
import numpy as np
import time
import copy

# Scipy
from scipy import linalg

import matplotlib.pyplot as plt

# Pyspark
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import BlockMatrix
from pyspark.mllib.linalg.distributed import DenseMatrix



# %%
import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession
sc = SparkSession.builder.appName('myApp').getOrCreate()

# %%
data_type = 'float32'

# %%
def makeBlocks(arr, nrows, ncols):
    '''
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    '''

    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisible by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisible by {}".format(w, cols)

    return (arr.reshape(h//nrows, nrows, -1, ncols)
          .swapaxes(1, 2)
          .reshape(-1, nrows, ncols))

# %%
def MapToPair(block, size):
    ri = block[0][0]
    ci = block[0][1]

    if (ri//size == 0) and (ci//size == 0):
        tag = "A11"
    elif (ri//size == 0) and (ci//size == 1):
        tag = "A12"
    elif (ri//size == 1) and (ci//size == 0):
        tag = "A21"
    elif (ri//size == 1) and (ci//size == 1):
        tag = "A22"

    # Get the number of rows, number of columns and values to create a new block
    numRows = block[1].numRows
    numCols = block[1].numCols
    matrixValues = block[1].values

    # Row Index and Column Index of the new block
    rowIndex = ri % size
    colIndex = ci % size

    newBlock = ((rowIndex, colIndex), Matrices.dense(numRows, numCols, matrixValues))

    return (tag, newBlock)

# %%
def breakMat(A, size):
    ARDD = A.blocks
    return ARDD.map(lambda x: MapToPair(x, size))

# %%
def function_xy(x, y, pairRDD, block_size):
    tag = 'A' + x + y
    filteredRDD = pairRDD.filter(lambda x: x[0] == tag)
    blocks = filteredRDD.map(lambda x: x[1])
    return BlockMatrix(blocks, block_size, block_size)

# %%
def scipy_inverse(block):
    # Get the Row Index and Column Index of the block
    rowIndex = block[0][0]
    colIndex = block[0][1]

    # Get values to find the inverse
    matrixValues = block[1].toArray()

    # Find inverse using scipy
    inverse_matrixValues = linalg.inv(matrixValues)

    # Change the inverse matrix to column major order
    inverse_matrixValues = inverse_matrixValues.flatten(order='F')

    inverse_block = ((rowIndex, colIndex), Matrices.dense(block[1].numRows, block[1].numCols, inverse_matrixValues))

    return inverse_block

# %%
def multiply(mat1, mat2):
    mat1_mat2 = mat1.multiply(mat2)
    return mat1_mat2

# %%
def subtract(mat1, mat2):
    mat1_mat2 = mat1.subtract(mat2)
    return mat1_mat2

# %%
def scalarMulHelper(block, scalar):
    # Get the RowIndex and the ColIndex of the block
    rowIndex = block[0][0]
    colIndex = block[0][1]

    # Get values to multiply with a scalar
    matrixValues = block[1].values
    matrixValues = matrixValues*scalar

    newBlock = ((rowIndex, colIndex), Matrices.dense(block[1].numRows, block[1].numCols, matrixValues))

    return newBlock

# %%
def scalarMul(A, scalar, block_size):
    ARDD = A.blocks
    blocks = ARDD.map(lambda x: scalarMulHelper(x, scalar))
    return BlockMatrix(blocks, block_size, block_size)

# %%
def map_c12(block, size):
    # Get the RowIndex and the ColIndex of the block
    rowIndex = block[0][0]
    colIndex = block[0][1]
    colIndex = colIndex + size
    return ((rowIndex, colIndex), Matrices.dense(block[1].numRows, block[1].numCols, block[1].values))

def map_c21(block, size):
    # Get the RowIndex and the ColIndex of the block
    rowIndex = block[0][0]
    rowIndex = rowIndex + size
    colIndex = block[0][1]
    return ((rowIndex, colIndex), Matrices.dense(block[1].numRows, block[1].numCols, block[1].values))

def map_c22(block, size):
    # Get the RowIndex and the ColIndex of the block
    rowIndex = block[0][0]
    rowIndex = rowIndex + size
    colIndex = block[0][1]
    colIndex = colIndex + size
    return ((rowIndex, colIndex), Matrices.dense(block[1].numRows, block[1].numCols, block[1].values))

# %%
def arrange(C11, C12, C21, C22, size, block_size):
    C11RDD = C11.blocks
    C12RDD = C12.blocks
    C21RDD = C21.blocks
    C22RDD = C22.blocks

    C1 = C12RDD.map(lambda x: map_c12(x, size//block_size))
    C2 = C21RDD.map(lambda x: map_c21(x, size//block_size))
    C3 = C22RDD.map(lambda x: map_c22(x, size//block_size))

    unionRDD = C11RDD.union(C1.union(C2.union(C3)))

    return BlockMatrix(unionRDD, block_size, block_size)

# %%
def inverse(A, size, block_size):
    n = size//block_size
    if n == 1:
        A_RDD = A.blocks
        A_Inverse_Block = A_RDD.map(lambda x: scipy_inverse(x))
        return BlockMatrix(A_Inverse_Block, block_size, block_size)
    else:
        size = size/2
        pairRDD = breakMat(A, size//block_size)
        A11 = function_xy(str(1), str(1), pairRDD, block_size)
        A12 = function_xy(str(1), str(2), pairRDD, block_size)
        A21 = function_xy(str(2), str(1), pairRDD, block_size)
        A22 = function_xy(str(2), str(2), pairRDD, block_size)
        one = inverse(A11, size, block_size)
        two = multiply(A21, one)
        three = multiply(one, A12)
        four = multiply(A21, three)
        five = subtract(four, A22)
        six = inverse(five, size, block_size)
        C12 = multiply(three, six)
        C21 = multiply(six, two)
        seven = multiply(three, C21)
        C11 = subtract(one, seven)
        C22 = scalarMul(six, -1, block_size)
        C = arrange(C11, C12, C21, C22, size, block_size)
        return C

# %%
def calculate_time_duration(A, matrix_size, block_size):
    # Assuming numpy array as input
    start_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))

    # Assuming numpy array as input
    start_time = time.time()
    input_arr_scipy_inverse = linalg.inv(input_arr)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    A_inv = inverse(A, matrix_size, block_size)
    print("--- %s seconds ---" % (time.time() - start_time))

    return (time.time() - start_time)




# %%
block_sizes = [16, 32, 64, 128]

matrix_size = 256


# %%
sc.sparkContext._conf.set("spark.executor.heartbeatInterval", "1000000000000000").set("spark.network.timeout", "1000000000000000")

#%%
# Calculate time duration for different block sizes

time_duration = []

input_arr = np.random.rand(matrix_size, matrix_size).astype(data_type)

for block_size in block_sizes:
    print("Calculating for block: "+str(block_size))

    block_arrays = makeBlocks(input_arr, block_size, block_size)
    print("Shape of Blocked Array is: {}".format(block_arrays.shape))
    # print("Blocked Array is: {}".format(block_arrays))

    block_arrays_list = []
    num_blocks = block_arrays.shape[0]
    num_rowIndex = input_arr.shape[0]//block_size
    for idx in range(num_blocks):
        block  = ((idx//num_rowIndex, idx%num_rowIndex), Matrices.dense(block_size, block_size, block_arrays[idx].flatten(order='F')))
        block_arrays_list.append(block)

    # print("Blocked Array List is: {}".format(block_arrays_list))

    # Parallelize the Block Array List
    blocks = sc.sparkContext.parallelize(block_arrays_list)

    # Type will be .rdd
    # print("Type of blocks is: {}".format(type(blocks)))

    A = BlockMatrix(blocks, block_size, block_size)

    time_taken = calculate_time_duration(A, matrix_size, block_size)

    time_duration.append(time_taken)

print(time_duration)

# plot the time duration for different block sizes

plt.plot(block_sizes, time_duration)
plt.scatter(block_sizes, time_duration)

plt.xticks(block_sizes, block_sizes)

plt.xlabel('Block Size')
plt.ylabel('Time Duration')
plt.title('For matrix_size ' + str(matrix_size) +': Time Duration vs Block Size')
plt.show()



# %%



