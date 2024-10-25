import numpy as np
import random
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
def check_column_zeros(matrix, column_index):
    num_rows = len(matrix)
    # Start iterating from the second row (index 1)
    for row in range(1, num_rows):
        if matrix[row][column_index] != 0:
            return False  # If any element is not 0, return False

    return True  # All elements in the column are 0
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
                max_endtime=0
                for sublist in lists:
                        for elem in  lists[sublist]:
                            if elem.name in list_prereq_must:
                                #print('elem.name in must: '+str(elem.name))
                                end_time = elem.end_time
                                if end_time>max_endtime:
                                    max_endtime= end_time
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
def check_columns_same(matrix, index1, index2):
    # Iterate through the rows of the matrix
    for row in matrix:
        # Compare elements in columns index1 and index2
        if row[index1] != row[index2]:
            return False  # Columns are not the same
    return True
def assign_machine (list_solution, machine_array, original_task):
    #print('assign machine: ')
    list_solution = [elem for elem in list_solution if elem <= original_task]

    n = len(machine_array)
    list_v2 = [0] * n
    for i in range(n):
        for j in range(len(list_solution)):
            if i+1 == list_solution[j]:
                list_v2[j]=array_machine[i]

    v1v2= [list_solution,list_v2]
    print(v1v2)
    return v1v2  
def mutation_jobshop(array_arrangement, matrix_output):
    i = random.randint(1, len(array_arrangement))
    j = random.randint(1, len(array_arrangement))
    while i == j:
        j = random.randint(1, len(array_arrangement))
    if check_columns_same(matrix_output, i, j):
        index_of_i = array_arrangement.index(i)+1
        print('i is:'+str(i)+", indext of i is:"+str(index_of_i))
        index_of_j = array_arrangement.index(j)+1
        print('j is:'+str(j)+", indext of j is:"+str(index_of_j))
        if index_of_i < index_of_j:
            array_arrangement.remove(j)
            array_arrangement.insert(index_of_i-1, j)
        else:
            array_arrangement.remove(i)
            array_arrangement.insert(index_of_j-1, i)
        return array_arrangement
    else:
        return mutation_jobshop(array_arrangement, matrix_output)
                 

test_array= [3, 9, 2, 8, 1, 10, 7, 6, 5, 4]
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

matrix= np.array([
    [None, None, None, None, None, None, None, None, None, None,None,None],  # row 0
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],  # row 1
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, -1],  # row 2
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, -1],  # row 3
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],  # row 4
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],  # row 5
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],  # row 6
    [None, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,0],  # row 7
    [None, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,0],  # row 8
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],  # row 9
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],  # row 10
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
], dtype=object)

matrix_time = np.array([
    [None, 1,2,3,4,5,6,7,8,9,10],  # row 0
    [None, 14,10,12,18,23,16,20,36,14,10],  # row 1
], dtype=object)
array_machine = [1,2,3,1,2,3,1,2,3,1]
list_solutions=[]
for i in range(15):
    print(i)
    mutation = mutation_jobshop(test_array,matrix)
    mutation_with_machines= assign_machine(mutation, array_machine, 10)
    list_solutions.append(mutation_with_machines)
print(list_solutions)


