import numpy as np
import random
import matplotlib.pyplot as plt
import math
def are_all_elements_identical(array):
    # Return True for empty arrays
    if not array:
        return True

    # Check if every element is identical to the first one
    first_element = array[0]
    return all(element == first_element for element in array)
def find_largest_element(array):
    if not array:
        return None, None  # Return None if the array is empty

    max_element = max(array)
    index_of_max = array.index(max_element)

    return max_element, index_of_max

def check_column_zeros(matrix, column_index):
    num_rows = len(matrix)
    # Start iterating from the second row (index 1)
    for row in range(1, num_rows):
        if matrix[row][column_index] != 0:
            return False  # If any element is not 0, return False

    return True  # All elements in the column are 0


def column_same(matrix, c1, c2):
    matrix_np = np.array(matrix, dtype=object)  # Convert the list of lists to a NumPy array
    c1_column = matrix_np[:, c1]
    c2_column = matrix_np[:, c2]
    columns_are_same = np.array_equal(c1_column, c2_column)
    return columns_are_same


def add_row_with_zeros(matrix):
    n, m = matrix.shape  # Get the current number of rows (n) and columns (m)
    # Create a new row filled with zeros
    new_row = np.zeros((1, m), dtype=matrix.dtype)
    # Use np.vstack to stack the original matrix and the new row vertically
    new_matrix = np.vstack((matrix, new_row))
    return new_matrix

def copy_and_add_column(matrix, column_index):
    # Copy the specified column from the matrix
    copied_column = matrix[:, column_index:column_index + 1].copy()
    # Add the copied column as a new column on the right
    matrix_with_added_column = np.hstack((matrix, copied_column))
    return matrix_with_added_column
    

def check_colum_first_n_elem (matrix,index1,index2, n):
    for i in range(1,n+1):
        if matrix [i][index1]== matrix [i][index2]:
            return True
        else:
            return False
        
def reset(matrix):
    original_size= len(matrix)-1
    #print('original size'+str(original_size))
    for i in range (1,original_size+1):
        for j in range(1,original_size+1):
            if matrix[i][j]==-1:
                #print((i,j))
                #if need to extend the matrix
                if len(matrix)==original_size+1 or check_colum_first_n_elem(matrix,j,-1,original_size)== False:
                    matrix=copy_and_add_column(matrix,j)
                    matrix=add_row_with_zeros(matrix)
                    matrix[-1][j]=1
                    #print('this is ij:'+str((i,j)))
                    
                elif check_colum_first_n_elem(matrix,j,-1,original_size)== True:
                    #print('HERE'+str((i,j)))
                    matrix[-1][j]=1
                matrix[i][j]=0
    return matrix


def set_initial_info (matrix_reset):
    column=len(matrix_reset) #the first is none
    row = 2 # The fourth row is for the result
    matrix_output = [[[] for _ in range(column)] for _ in range(row)]
    matrix_output[1] = [1] * column #set the second row to 1
    start= []
    
    for i in range(1,len(matrix_reset)):
       # print('here'+str(i))
        if check_column_zeros(matrix_reset,i)==True:
            start.append(i)
    for elem in start:
        matrix_output[0][elem]=None
        matrix_output[1][elem]=0
    for i in range(1,len(matrix_reset)): # i is column
        for j in range(1,len(matrix_reset)): # j is row
            if matrix_reset[j][i]==1:
                #print('here is matrix reset ji=1:'+str(j)+','+str(i))
                matrix_output[0][i]= j
                #print('matrix out put [0][i]:'+str(i)+' , i ='+str(j))
            if matrix_reset[j][i]==-1: 
                matrix_output[0][i].append(j)
    return matrix_output

def check_row (matrix_ini_info,i):
    row_to_check = matrix_ini_info[i]
    if all(element == -1 for element in row_to_check[1:]):
        return True
    else:
       return False


#table 2 and 3
def find_solution(matrix_ini_info_original):
    matrix_ini_info=matrix_ini_info_original
    list_solution=[]
    while check_row(matrix_ini_info,1)==False:
        random_number = random.randint(1, len(matrix_ini_info[1])-1) #the whole interval is inclusive
        if matrix_ini_info[1][random_number] ==0:
            list_solution.append(random_number)
           # print('problem!!!'+str(random_number))
            matrix_ini_info[1][random_number]=-1

            #check which elem in row 1 contains this selected task
            for j in range(len(matrix_ini_info[0])):
            # Check if the element is a list
                if isinstance(matrix_ini_info[0][j], list):
                # Iterate through elements within the inner list
                    for element in matrix_ini_info[0][j]:
                    # Perform your check here for each element
                        if element== random_number :
                            #print(f"Element {element} meets the condition.*list")
                            if matrix_ini_info[1][j] ==1: #ensure that it is not been done yet
                                matrix_ini_info[1][j]=0
                                #print('new ready:'+str(j))
                            #print(check_row(matrix_ini_info,1))
                        #else:
                            # print(f"Element {element} does not meet the condition.")
                else: # if it is not list
                # Perform your check for non-list elements here
                    if matrix_ini_info[0][j]==random_number :
                        #print(f"Element {matrix_ini[0][j]} meets the condition.*not list")
                        if matrix_ini_info[1][j] ==1:
                            matrix_ini_info[1][j]=0
                           # print('new ready:'+str(j))
                        #print(check_row(matrix_ini_info,1))
    #print('matrix_ini_info_original')
    #print(matrix_ini_info_original)
    return list_solution  #v1 list

def assign_machine (list_solution, num_m, original_task):
    #print('assign machine: ')
    list_v2=[]
    list_solution = [elem for elem in list_solution if elem <= original_task]
    
    for elem in list_solution:
        #print('here is list solution reset'+str(elem))
        random_num= random.randint(1,num_m)
        print('randonnum: '+str(random_num))
        list_v2.append(random_num)
    print('listv2:'+str(list_v2))
    v1v2= [list_solution,list_v2]
    return v1v2   #this does not have dume start, and only contains the real tasks


#for calculation of time and gantt chart
class Machine:
    def __init__(self, name, tasks=None):
        self.name = name
        self.tasks = tasks or []

    def add_task(self, name, start_time, end_time):
        self.tasks.append(Task(name, start_time, end_time))

class Task:
    def __init__(self, name, start_time, end_time):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time


def time_cal(time_matrix, v1v2, original_matrix,v2_num): #v2_num is the machine number
    finished_task=[]
    # create v2_num machines
    lists = {i: [] for i in range(v2_num)}
    for i in range (len(v1v2[0])):
        #print('this is loop #'+str(i+1))
        mytask= Task(None,None,None)
        mytask.name= v1v2[0][i]
        #print('This is task :'+str(mytask.name))
        machine= v1v2[1][i]
        duration = time_matrix[1][mytask.name]

        new_begin_time=1000000
        if check_column_zeros(original_matrix, mytask.name) == False: #preatep is neccessary
                list_prereq_must= []  #and op
                list_prereq_or=[]    # or op
                for i in range(1,len(original_matrix)):
                    if original_matrix[i][mytask.name]==-1:
                        list_prereq_or.append(i)
                    elif original_matrix[i][mytask.name]==1:
                        list_prereq_must.append(i)
                #print('list_prereq_must: ')
                #for elm1 in list_prereq_must:
                    #print(elm1)
                #print('list_prereq_or: ')
                #for elm2 in list_prereq_or:
                    #print(elm2)
                #result_must = all(elem in list_prereq_must for elem in finished_task)
                #print('result must: '+ str(result_must))
                #result_or = set(list_prereq_or).intersection(finished_task)#problem
                #print('result or: '+ str(result_or))
                #print('here!!!do not go inside here')               
                #if result_must or result_or:      #????????
                max_endtime=0
                    #print('!!do not go inside here')
                    #check must_list, logic is all
                
                for sublist in lists:
                        for elem in  lists[sublist]:
                            if elem.name in list_prereq_must:
                                #print('elem.name in must: '+str(elem.name))
                                end_time = elem.end_time
                                #print('elem.endtime in list_prereq_must:')
                                #print(end_time)
                                #get the latest one from the must list
                                if end_time>max_endtime:
                                    max_endtime= end_time
                                    
                    #check or_list
                min_endtime=0
                for sublist in lists:
                        for task in lists[sublist]:
                            #print('hdo not go inside here')
                            if task.name in list_prereq_or:
                                if min_endtime==0:
                                   min_endtime= task.end_time
                                   
                                else:
                                    if task.end_time < min_endtime:
                                        min_endtime =task.end_time
                                #print('min_endtime: '+ str(min_endtime))
                maxtime_prereq= max(max_endtime,min_endtime)
                non_prereq_endtime=0
                if  lists[machine-1]:
                    non_prereq_endtime = lists[machine-1][-1].end_time
                new_begin_time=max(maxtime_prereq,non_prereq_endtime)
                #print('new begin time: '+str(new_begin_time))
        else:  # no need for prestep
                #print('!do not go inside here')
                if lists[machine-1]:  #if the machine is not empty
                    new_begin_time = lists[machine-1][-1].end_time
                else:
                    new_begin_time=0
                    mytask.end_time = time_matrix[1][mytask.name]
                


        mytask.start_time= new_begin_time
        mytask.end_time= mytask.start_time + duration
            #print('here!!!do not go inside here')
        lists[machine-1].append(mytask)
        finished_task.append(mytask.name)
        
    return lists

def draw_ganttchart(ganttchart):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the Gantt chart bars for tasks and sub-tasks
    for i, (task_name, sub_tasks) in enumerate(ganttchart):
        for j, (sub_task_name, start_date, end_date) in enumerate(sub_tasks):
            y_coord = i  # Adjust the y-coordinate to be the same as the parent task
            ax.barh(
                y_coord,
                width=end_date - start_date,
                left=start_date,
                label=f"{task_name} - {sub_task_name}"
            )

    # Beautify the plot
    plt.yticks(range(len(ganttchart)), [task_name for task_name, _ in ganttchart])  # Set y-axis labels to task names
    plt.xlabel("Timeline")
    plt.ylabel("Tasks")
    plt.title("Gantt Chart with Sub-Tasks")

    # Add a legend
    plt.legend(loc="upper left")

    # Display the Gantt chart
    plt.tight_layout()
    plt.show()


def maxtime(listoflists):
    #print('Max time of 3 machines: ')
    max_time=0
    for list in listoflists.values():
        for elem in list:
            #print(str(elem.name)+';'+str(elem.start_time)+';'+str(elem.end_time))
            if elem.end_time>max_time:
                max_time=elem.end_time
    return max_time

def mutation(v1v2,m_num):  
    list = assign_machine(v1v2[0],m_num, len(v1v2[0]))
    return list
def crossover(l1,l2):
    l1_a1=[]
    a2=[]
    for i in l1:
        l1_a1.append(random.randint(1, 2))
        a2.append(random.randint(1, 2))
    l1copy = l1[:]
    l2copy = l2[:]
    lb1=[]
    for i in range(len(l1)):
        if l1_a1[i]==1 :
            if l1copy[0] not in lb1:
                lb1.append(l1copy[0])
                l1copy.pop(0)
            elif l1copy[0] in lb1:
                l1copy= [element for element in l1copy if element not in lb1]
                lb1.append(l1copy[0])
        else: 
            if l2copy[0] not in lb1:
                lb1.append(l2copy[0])
                l2copy.pop(0)
            elif l2copy[0] in lb1:
                l2copy= [element for element in l2copy if element not in lb1]
                lb1.append(l2copy[0])

    l1copy = l1[:]
    l2copy = l2[:]
    lb2=[]
    for i in range(len(l1)):
        if a2[i]==1 :
            if l1copy[0] not in lb2:
                lb2.append(l1copy[0])
                l1copy.pop(0)
            elif l1copy[0] in lb2:
                l1copy= [element for element in l1copy if element not in lb2]
                lb2.append(l1copy[0])
        else: 
            if l2copy[0] not in lb2:
                lb2.append(l2copy[0])
                l2copy.pop(0)
            elif l2copy[0] in lb2:
                l2copy= [element for element in l2copy if element not in lb2]
                lb2.append(l2copy[0])
    return lb1,lb2

def softmax(matrix):
    int_list=matrix[0]
    # Custom transformation to assign probabilities
    transformed_int_list = [max(int_list) - x for x in int_list]

    # Set a temperature parameter
    temperature = 11

    # Apply softmax function to the transformed list with adjusted temperature
    probabilities = np.exp(np.array(transformed_int_list) / temperature) / np.sum(np.exp(np.array(transformed_int_list) / temperature))

    # Print the probabilities of each integer
    #for i, num in enumerate(int_list):
       # print(f"Probability of {num}: {probabilities[i]:.3f}")

    # Select an integer based on the computed probabilities
    selected_int = np.random.choice(int_list, p=probabilities)

    # Select an integer based on the computed probabilities
    selected_int = np.random.choice(int_list, p=probabilities)
    selected_index = np.where(int_list == selected_int)[0]
   
    #print(f"\nSelected integer: {selected_int}")
    #print(f"Index of selected integer: {selected_index}")
    return selected_int,selected_index[0]

matrix_p = np.array([
    [None, None, None, None, None, None, None, None, None, None,None],  # row 0
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 1
    [None, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1],  # row 2
    [None, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1],  # row 3
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 4
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 5
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 6
    [None, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # row 7
    [None, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # row 8
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 9
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 10
], dtype=object)
   
matrix_time = np.array([
    [None, 1,2,3,4,5,6,7,8,9,10],  # row 0
    [None, 14,10,12,18,23,16,20,36,14,10],  # row 1
], dtype=object)
listoffinalbabies_time=[]
listoffinalbabies_solution=[]



# get some solutions
list_ga=[]
for i in range(6):
    matrix_reset= reset(matrix_p)
    matrix_ini =set_initial_info(matrix_reset)
    #print(matrix_ini)
    inimatrix=matrix_ini
    list_so= find_solution(matrix_ini)
    v1andv2= assign_machine(list_so,3,10)
    list_ga.append(v1andv2)
list_time=[]
for i in range(6):
        ind=i
        listoflists_2=time_cal(matrix_time,list_ga[ind],matrix_p,3)
        list_time.append(maxtime(listoflists_2))
#print(list_time)
#print('listga5')
#for i in range(5):
   # print(list_ga[i])

#draw ganttchart
ganttchart=  [(f"M {i+1}", [(task.name, task.start_time, task.end_time) for task in sublist]) for i, sublist in listoflists_2.items()]
#print('Time is: '+str(max_time))
print(ganttchart)
draw_ganttchart(ganttchart)





array_of_int = np.array(list_time)

    # Get the indices of the top 5 smallest elements
indices_of_top_5_smallest = np.argsort(array_of_int)[:5]

    # Extract the top 5 smallest elements and their indices
top_5_smallest = array_of_int[indices_of_top_5_smallest]
indices_array = np.array(indices_of_top_5_smallest)
#print('top5')
#print(top_5_smallest)
    # Create the desired matrix
time_and_index = np.array([top_5_smallest, indices_array])

    # Print the resulting matrix
#print("time_and_index:")
#print(time_and_index)

def update_solution_list(listoflist,n):
    if n==0:
        return listoflist
    else:
        indices_parents=[]
        for i in range(4):
            selection,index=softmax(time_and_index)
            indices_parents.append(index)
        #print('indices_parents:')
        list_selected_solution=[]
        for elem in time_and_index[1]:
                list_selected_solution.append(list_ga[elem])
        papa1=list_selected_solution[indices_parents[0]]
        mama1=list_selected_solution[indices_parents[1]]
       # print('mama1')
       # print(mama1)
       # print('papa1')
       # print(papa1)
        baby1,baby2=crossover(papa1[0],mama1[0])
       # print('baby1')
       # print(baby1)
        baby1_m=assign_machine(baby1,3,len(baby1))
       # print(baby1_m)
       # print('baby2')
       # print(baby2)
        baby2_m=assign_machine(baby2,3,len(baby2))
        babylist=[]
        babylist.append(baby1_m)
        babylist.append(baby2_m)
        papa2=list_selected_solution[indices_parents[2]]
        mama2=list_selected_solution[indices_parents[3]]

        baby3,baby4=crossover(papa2[0],mama2[0])
       # print('baby3')
       # print(baby3)
        baby3_m=assign_machine(baby3,3,len(baby3))
        #print(baby3_m)
       # print('baby4')
       # print(baby4)
        baby4_m=assign_machine(baby4,3,len(baby4))
        babylist.append(baby4_m)
        babylist.append(baby4_m)
        #print(baby4_m)

        list_time_new=[]
        for i in range(len(babylist)):
                ind=i
                listoflists_21=time_cal(matrix_time,babylist[ind],matrix_p,3)
                list_time_new.append(maxtime(listoflists_21))
        #print(list_time_new)
        for elem in list_time_new:
            listoffinalbabies_time.append(elem)
        for elem in babylist:
            listoffinalbabies_solution.append(elem)
        new_solution=listoffinalbabies_solution
        original_time=[]
        for i in range(len(listoflist)):
            
            time=time_cal(matrix_time,listoflist[i],matrix_p,3)
            original_time.append(maxtime(time))
        print('orgina time')
        print(original_time)
        print('new time')
        print(list_time_new)


        max_index = max(range(len(original_time)), key=lambda i: original_time[i])

# Find index of the smallest value in list_time_new
        min_index = min(range(len(list_time_new)), key=lambda i: list_time_new[i])
        # Replace values in original_tim with values from list_time_new
        if original_time[max_index] > list_time_new[min_index]:
            original_time[max_index] = list_time_new[min_index]
            listoflist[max_index]=new_solution[min_index]
            #print(original_time)
       
        #print(f"Updated list after iteration {n}: {listoflist}")
        print(f"time of {n}: {min(original_time)}")
       # print(f"time of {n}: {original_time}")
        return update_solution_list(listoflist,n-1)
        

listsolution=update_solution_list(list_ga,20)
print(listsolution)

