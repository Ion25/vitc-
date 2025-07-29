#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "matrix.h"

// Funciones de activación
float relu(float x);
float gelu(float x);
float gelu_derivative(float x);
float sigmoid(float x);
float tanh_activation(float x);

// Función para aplicar GELU a matriz completa
Matrix apply_gelu(const Matrix& input);

#endif // ACTIVATIONS_H