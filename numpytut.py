import numpy as np

# ndarray is the basic numpy building block

#generates an ndarray
#np.arange([start,] stop [,step,], dtype=None)
# eg: 'a range', not 'arange'...
print(np.arange(10))

print(np.arange(1, 10))

print(np.arange(1, 10, 0.5))

print(np.arange(0, 10, 1, dtype=np.float64))


ds = np.arange(1, 10, 2)

#number of dimensions
print(ds.ndim)

print(ds.shape)

print(ds.size)

x = ds.data
print(list(x))


#memory usage
print( ds.size * ds.itemsize)


#Generate a random array of numbers between 0 and 1
data_set = np.random.random( (2, 3) )

print(data_set)
print(np.max(data_set))
print(np.min(data_set))
print(np.mean(data_set))

print(np.max(data_set, axis=0))
print(np.max(data_set, axis=1))

print("Sum")
print(np.sum(data_set))
print("Column sum:")
print(np.sum(data_set, axis=0))
print("Row sum:")
print(np.sum(data_set, axis=1))


#Reshaping
data_set = np.reshape(data_set, (3, 2))
print(data_set)
data_set = np.reshape(data_set, (6, 1))
print(data_set)
data_set = np.reshape(data_set, (6))
print(data_set)


#Slicing
data_set = np.random.random( (3, 4) )
print(" ")
print(data_set)
print(" ")
print(data_set[1]) # 1th row
print(data_set[1][0]) #1th row, 0th column
print(data_set[1, 0]) #1th row, 0th Column
print(data_set[0, 1:3]) #0th row, 1th and 2th elements
print(data_set[:, 0]) #all items first dimension, 0th item second dimension (ie, the first element of each row)
print (" ")
print(data_set[-1, :]) #All elements from the last row
print(data_set[0:-1, :]) #All elements, all rows

print(data_set[2:3, ::2]) #Last row, every second value starting from 0
