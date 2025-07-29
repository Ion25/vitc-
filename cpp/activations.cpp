#include "activations.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float relu(float x) {
    return std::max(0.0f, x);
}

float gelu(float x) {
    const float sqrt_2_pi = std::sqrt(2.0f / M_PI);
    return 0.5f * x * (1.0f + std::tanh(sqrt_2_pi * (x + 0.044715f * x * x * x)));
}

float gelu_derivative(float x) {
    const float sqrt_2_pi = std::sqrt(2.0f / M_PI);
    const float tanh_arg = sqrt_2_pi * (x + 0.044715f * x * x * x);
    const float tanh_val = std::tanh(tanh_arg);
    const float sech_sq = 1.0f - tanh_val * tanh_val;
    
    return 0.5f * (1.0f + tanh_val) + 
           0.5f * x * sech_sq * sqrt_2_pi * (1.0f + 3.0f * 0.044715f * x * x);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float tanh_activation(float x) {
    return std::tanh(x);
}

Matrix apply_gelu(const Matrix& input) {
    Matrix result(input.rows, input.cols);
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            result.data[i][j] = gelu(input.data[i][j]);
        }
    }
    return result;
}