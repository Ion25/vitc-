#include "adam_optimizer.h"
#include <cmath>
#include <algorithm>

AdamOptimizer::AdamOptimizer(float lr, float b1, float b2, float eps) 
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0), initialized(false) {
}

void AdamOptimizer::initialize(int num_weight_matrices, int num_bias_matrices) {
    if (initialized) return;
    
    m_weights.clear();
    v_weights.clear();
    m_biases.clear();
    v_biases.clear();
    
    m_weights.reserve(num_weight_matrices);
    v_weights.reserve(num_weight_matrices);
    m_biases.reserve(num_bias_matrices);
    v_biases.reserve(num_bias_matrices);
    
    for (int i = 0; i < num_weight_matrices; i++) {
        m_weights.emplace_back(1, 1); 
        v_weights.emplace_back(1, 1);  
    }
    
    for (int i = 0; i < num_bias_matrices; i++) {
        m_biases.emplace_back(1, 1);  
        v_biases.emplace_back(1, 1); 
    }
    
    initialized = true;
}

void AdamOptimizer::updateWeights(Matrix& weights, const Matrix& gradients, int param_index) {
    if (!initialized || param_index >= (int)m_weights.size()) {
        return; 
    }
    
    if (m_weights[param_index].rows != weights.rows || m_weights[param_index].cols != weights.cols) {
        m_weights[param_index] = Matrix(weights.rows, weights.cols);
        v_weights[param_index] = Matrix(weights.rows, weights.cols);
        
        for (int i = 0; i < weights.rows; i++) {
            for (int j = 0; j < weights.cols; j++) {
                m_weights[param_index].data[i][j] = 0.0f;
                v_weights[param_index].data[i][j] = 0.0f;
            }
        }
    }
    
    // Adam 
    for (int i = 0; i < weights.rows; i++) {
        for (int j = 0; j < weights.cols; j++) {
            float grad = gradients.data[i][j];
            
            m_weights[param_index].data[i][j] = beta1 * m_weights[param_index].data[i][j] + (1.0f - beta1) * grad;
            
            v_weights[param_index].data[i][j] = beta2 * v_weights[param_index].data[i][j] + (1.0f - beta2) * grad * grad;
            
            float m_hat = m_weights[param_index].data[i][j] / (1.0f - std::pow(beta1, t + 1));
            float v_hat = v_weights[param_index].data[i][j] / (1.0f - std::pow(beta2, t + 1));
            
            weights.data[i][j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);

        }
    }
}

void AdamOptimizer::updateBias(Matrix& bias, const Matrix& gradients, int param_index) {
    if (!initialized || param_index >= (int)m_biases.size()) {
        return;  
    }
    
    if (m_biases[param_index].rows != bias.rows || m_biases[param_index].cols != bias.cols) {
        m_biases[param_index] = Matrix(bias.rows, bias.cols);
        v_biases[param_index] = Matrix(bias.rows, bias.cols);
        
        for (int i = 0; i < bias.rows; i++) {
            for (int j = 0; j < bias.cols; j++) {
                m_biases[param_index].data[i][j] = 0.0f;
                v_biases[param_index].data[i][j] = 0.0f;
            }
        }
    }
    
    // Adam algorithm
    for (int i = 0; i < bias.rows; i++) {
        for (int j = 0; j < bias.cols; j++) {
            float grad = gradients.data[i][j];
            
            m_biases[param_index].data[i][j] = beta1 * m_biases[param_index].data[i][j] + (1.0f - beta1) * grad;
            
            v_biases[param_index].data[i][j] = beta2 * v_biases[param_index].data[i][j] + (1.0f - beta2) * grad * grad;
            
            float m_hat = m_biases[param_index].data[i][j] / (1.0f - std::pow(beta1, t + 1));
            float v_hat = v_biases[param_index].data[i][j] / (1.0f - std::pow(beta2, t + 1));
            
            bias.data[i][j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            float weight_decay = 0.005f;
            bias.data[i][j] *= (1.0f - learning_rate * weight_decay);
        }
    }
}

void AdamOptimizer::step() {
    t++;  
}

void AdamOptimizer::reset() {
    t = 0;
    
    for (auto& m : m_weights) {
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                m.data[i][j] = 0.0f;
            }
        }
    }
    
    for (auto& v : v_weights) {
        for (int i = 0; i < v.rows; i++) {
            for (int j = 0; j < v.cols; j++) {
                v.data[i][j] = 0.0f;
            }
        }
    }
    
    for (auto& m : m_biases) {
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                m.data[i][j] = 0.0f;
            }
        }
    }
    
    for (auto& v : v_biases) {
        for (int i = 0; i < v.rows; i++) {
            for (int j = 0; j < v.cols; j++) {
                v.data[i][j] = 0.0f;
            }
        }
    }
}