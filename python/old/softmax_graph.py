import numpy as np

def softmax(matrix):
    int_list=matrix[0]
    # Custom transformation to assign probabilities
    transformed_int_list = [max(int_list) - x for x in int_list]

    # Set a temperature parameter
    temperature = 11

    # Apply softmax function to the transformed list with adjusted temperature
    probabilities = np.exp(np.array(transformed_int_list) / temperature) / np.sum(np.exp(np.array(transformed_int_list) / temperature))

    # Print the probabilities of each integer
    for i, num in enumerate(int_list):
        print(f"Probability of {num}: {probabilities[i]:.3f}")

    # Select an integer based on the computed probabilities
    selected_int = np.random.choice(int_list, p=probabilities)

    # Select an integer based on the computed probabilities
    selected_int = np.random.choice(int_list, p=probabilities)
    selected_index = np.where(int_list == selected_int)[0]
   
    #print(f"\nSelected integer: {selected_int}")
    #print(f"Index of selected integer: {selected_index}")
    return selected_int,selected_index[0]
   
    return selected_int, selected_index[0]
matrix=  [[1,2,3,4,5,6],[677,587,874,786,811,782]]
print(softmax(matrix))

import numpy as np
import matplotlib.pyplot as plt

# Given probabilities
probabilities = [0.207, 0.189, 0.172, 0.157, 0.144, 0.131]
labels = ['677 sec', '587 sec', '874 sec', '786 sec', '811 sec', '782 sec']

# Define less saturated colors
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFD700', '#C0C0C0']

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(probabilities, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Probability Distribution')
plt.show()
