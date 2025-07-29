#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <string>
#include <iostream>

class Matrix {
public:
    std::vector<std::vector<float>> data;
    int rows, cols;
    
    Matrix(int r, int c);
    Matrix();
    
    // Inicializacion con distribucion normal
    void initializeRandom(float mean = 0.0f, float std = 0.1f);
    
    // Operaciones basicas
    Matrix multiply(const Matrix& other) const;
    Matrix add(const Matrix& other) const;
    Matrix transpose() const;
    Matrix apply(float (*func)(float)) const;
    Matrix softmax() const;
    Matrix slice(int start_row, int end_row, int start_col, int end_col) const;
    
    // Persistencia
    void saveToFile(const std::string& filename) const;
    bool loadFromFile(const std::string& filename);
};

#endif // MATRIX_H