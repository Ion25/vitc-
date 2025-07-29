#include "matrix.h"
#include <random>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>

Matrix::Matrix(int r, int c) : rows(r), cols(c) {
    data.resize(rows, std::vector<float>(cols, 0.0f));
}

Matrix::Matrix() : rows(0), cols(0) {}

void Matrix::initializeRandom(float mean, float std) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = dist(gen);
        }
    }
}

Matrix Matrix::multiply(const Matrix& other) const {
    if (cols != other.rows) {
        std::cout << "ERROR: Dimensiones incompatibles: " << rows << "x" << cols 
                  << " * " << other.rows << "x" << other.cols << std::endl;
    }
    assert(cols == other.rows);
    Matrix result(rows, other.cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            for (int k = 0; k < cols; k++) {
                result.data[i][j] += data[i][k] * other.data[k][j];
            }
        }
    }
    return result;
}

Matrix Matrix::add(const Matrix& other) const {
    assert(rows == other.rows && cols == other.cols);
    Matrix result(rows, cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::apply(float (*func)(float)) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = func(data[i][j]);
        }
    }
    return result;
}

Matrix Matrix::softmax() const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        float max_val = *std::max_element(data[i].begin(), data[i].end());
        float sum = 0.0f;
        
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = std::exp(data[i][j] - max_val);
            sum += result.data[i][j];
        }
        
        for (int j = 0; j < cols; j++) {
            result.data[i][j] /= sum;
        }
    }
    return result;
}

Matrix Matrix::slice(int start_row, int end_row, int start_col, int end_col) const {
    Matrix result(end_row - start_row, end_col - start_col);
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            result.data[i][j] = data[start_row + i][start_col + j];
        }
    }
    return result;
}

void Matrix::saveToFile(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file.write(reinterpret_cast<const char*>(&data[i][j]), sizeof(float));
        }
    }
}

bool Matrix::loadFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    
    data.resize(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file.read(reinterpret_cast<char*>(&data[i][j]), sizeof(float));
        }
    }
    return true;
}