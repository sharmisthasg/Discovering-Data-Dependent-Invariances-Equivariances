import numpy as np
import random

def generate_lines(num_samples, dataset_path, class_path, dim0, dim1):
    X = np.zeros((num_samples, dim0, dim1))
    for i in range(X.shape[0]):
        row = random.randint(0, dim0-1)
        start_col = random.randint(0, 7)
        for j in range(start_col, start_col+3):
            X[i, row, j] = 1
    np.save(dataset_path + "/" + class_path, X)

def generate_triangles(num_samples, dataset_path, class_path, dim0, dim1):
    X = np.zeros((num_samples, dim0, dim1))
    for i in range(X.shape[0]):
        row = random.randint(2, dim0-1)
        start_col = random.randint(0, 5)
        for j in range(start_col, start_col + 5):
            X[i, row, j] = 1
        for j in range(start_col+1, start_col+4):
            X[i, row-1, j] = 1
        for j in range(start_col+2, start_col+3):
            X[i, row-2, j] = 1
    np.save(dataset_path + "/" + class_path, X)

def generate_squares(num_samples, dataset_path, class_path, dim0, dim1):
    X = np.zeros((num_samples, dim0, dim1))
    for i in range(X.shape[0]):
        row = random.randint(2, dim0-1)
        col = random.randint(2, dim1-1)
        for j in range(row-2, row+1):
            for k in range(col-2, col+1):
                X[i, j, k] = 1
    np.save(dataset_path + "/" + class_path, X)

if __name__ == '__main__':
    num_samples = 10000
    num_classes = 3
    dataset_path = "2D_dataset"
    dim0 = 10
    dim1 = 10
    generate_lines(num_samples, dataset_path, "class_0", dim0, dim1)
    generate_triangles(num_samples, dataset_path, "class_1", dim0, dim1)
    generate_squares(num_samples, dataset_path, "class_2", dim0, dim1)
