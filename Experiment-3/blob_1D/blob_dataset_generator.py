import numpy as np
from random import randrange, choice
import math

def prev(index):
    i = 10
    if index == 0:
        return i-1
    else:
        return index - 1

def next(index):
    i = 10
    if index == i-1:
        return 0
    else:
        return index + 1

def main():
    n_data = 10000
    i = 10
    #class 1: one blob
    class_1 = np.zeros((n_data, i))
    for array in range(n_data):
        datapoint = np.zeros(i)
        index = randrange(i)
        datapoint[index] = 1
        class_1[array] = datapoint

    np.save('blob_dataset/1/class_1', class_1)
    y1 = np.load('blob_dataset/1/class_1.npy')
    print(y1)

    #class 2: two consecutive blobs
    class_2 = np.zeros((n_data, i))
    for array in range(n_data):
        datapoint = np.zeros(i)
        index = randrange(i)
        datapoint[index] = 1
        datapoint[next(index)] = 1
        class_2[array] = datapoint

    np.save('blob_dataset/2/class_2', class_2)
    y2 = np.load('blob_dataset/2/class_2.npy')
    print(y2)

    # class 3: three scattered blobs
    class_3 = np.zeros((n_data, i))
    for array in range(n_data):
        datapoint = np.zeros(i)
        indexlist = list(range(i))

        index = randrange(i)
        indexlist.remove(index)

        second_index = choice(indexlist)
        indexlist.remove(second_index)
        #exhaustive?
        if next(index) == second_index :
            indexlist.remove(prev(index))
            indexlist.remove(next(second_index))
        elif next(second_index) == index:
            indexlist.remove(prev(second_index))
            indexlist.remove(next(index))

        third_index = choice(indexlist)

        datapoint[index] = 1
        datapoint[second_index] = 1
        datapoint[third_index] = 1
        class_3[array] = datapoint

    np.save('blob_dataset/3/class_3', class_3)
    y3 = np.load('blob_dataset/3/class_3.npy')
    print(y3)

    # class 4: four consecutive blobs
    class_4 = np.zeros((n_data, i))
    for array in range(n_data):
        datapoint = np.zeros(i)
        index = randrange(i)
        datapoint[index] = 1
        datapoint[next(index)] = 1
        datapoint[next(next(index))] = 1
        datapoint[next(next(next(index)))] = 1
        class_4[array] = datapoint

    np.save('blob_dataset/4/class_4', class_4)
    y4 = np.load('blob_dataset/4/class_4.npy')
    print(y4)

    # class 5: five scattered blobs
    class_5 = np.zeros((n_data, i))
    for array in range(n_data):
        datapoint = np.zeros(i)
        indexlist = list(range(i))

        index = randrange(i)
        indexlist.remove(index)

        second_index = choice(indexlist)
        indexlist.remove(second_index)

        third_index = choice(indexlist)
        indexlist.remove(third_index)

        fourth_index = choice(indexlist)
        indexlist.remove(fourth_index)
        curr_indices = [index, second_index, third_index, fourth_index]

        #Case 1: 4 consecutive blobs with no extreme indices
        if max(curr_indices) - min(curr_indices) == 4:
            indexlist.remove(prev(min(curr_indices)))
            indexlist.remove(next(max(curr_indices)))

        #Case 2: 2 blobs each on both extremes
        if all(k in curr_indices for k in [0, 1, i-1, i-2]):
            indexlist.remove(i-3)
            indexlist.remove(2)

        #Case 3: 1-3 or 3-1 blobs on either extreme
        mini = min(curr_indices)
        maxi = max(curr_indices)
        #1-3
        if mini == 0 and all(k in curr_indices for k in [i-1, i-2, i-3]):
            indexlist.remove(i-4)
            indexlist.remove(1)
        #3-1
        if maxi == i-1 and all(k in curr_indices for k in [0, 1, 2]):
            indexlist.remove(i-2)
            indexlist.remove(3)

        fifth_index = choice(indexlist)

        datapoint[index] = 1
        datapoint[second_index] = 1
        datapoint[third_index] = 1
        datapoint[fourth_index] = 1
        datapoint[fifth_index] = 1
        class_5[array] = datapoint

    np.save('blob_dataset/5/class_5', class_5)
    y5 = np.load('blob_dataset/5/class_5.npy')
    print(y5)

if __name__ == '__main__':
    main()