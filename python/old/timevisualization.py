import numpy as np
from tabulate import tabulate

# Your matrix_time array
matrix_time = np.array([
    [None, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36],  
    [None, 49,64,36,12,210,24,10,0,24,0,24,48,60,0,0,48,50,48,24,0,0,0,48,5,0,10,72,36,24,20,10,70,10,10,10,10], 
], dtype=object)

# Extracting data
indices = matrix_time[0][1:]
times = matrix_time[1][1:]

# Combining data into a list of tuples
data = list(zip(indices, times))

# Table headers
headers = ['Index', 'Time (s)']

# Displaying the table
print(tabulate(data, headers=headers, tablefmt='grid'))
