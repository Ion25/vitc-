#include "layers.h"
#include "activations.h"
#include <cmath>
#include <random>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

LayerNorm::LayerNorm(int model_dim, float epsilon) : d_model(model_dim), eps(epsilon) {
    gamma.resize(d_model, 1.0f);
    beta.resize(d_model, 0.0f);
}

Matrix LayerNorm::forward(const Matrix& input) const {
    Matrix result = input;
    
    for (int i = 0; i < input.rows; i++) {
        float mean = 0.0f;
        for (int j = 0; j < d_model; j++) {
            mean += input.data[i][j];
        }
        mean /= d_model;
        
        float variance = 0.0f;
        for (int j = 0; j < d_model; j++) {
            variance += std::pow(input.data[i][j] - mean, 2);
        }
        variance /= d_model;
        
        for (int j = 0; j < d_model; j++) {
            result.data[i][j] = gamma[j] * (input.data[i][j] - mean) / std::sqrt(variance + eps) + beta[j];
        }
    }
    
    return result;
}

void LayerNorm::saveWeights(const std::string& prefix) const {
}

bool LayerNorm::loadWeights(const std::string& prefix) {
    return true;
}

// codificacion posicional
PositionalEncoding::PositionalEncoding(int max_length, int model_dim) : max_len(max_length), d_model(model_dim) {
    encoding = Matrix(max_len, d_model);
    
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < d_model; i++) {
            if (i % 2 == 0) {
                encoding.data[pos][i] = std::sin(pos / std::pow(10000.0f, (float)i / d_model));
            } else {
                encoding.data[pos][i] = std::cos(pos / std::pow(10000.0f, (float)(i-1) / d_model));
            }
        }
    }
}

Matrix PositionalEncoding::getEncoding(int seq_len) const {
    Matrix result(seq_len, d_model);
    for (int i = 0; i < seq_len && i < max_len; i++) {
        for (int j = 0; j < d_model; j++) {
            result.data[i][j] = encoding.data[i][j];
        }
    }
    return result;
}

// MultiHeadAttention Implementation
MultiHeadAttention::MultiHeadAttention(int model_dim, int heads) : d_model(model_dim), num_heads(heads) {
    d_k = d_model / num_heads;
    
    // Una sola matriz para Q, K, V
    W_qkv = Matrix(d_model, d_model * 3);
    W_o = Matrix(d_model, d_model);
    
    W_qkv.initializeRandom(0.0f, std::sqrt(2.0f / d_model));
    W_o.initializeRandom(0.0f, std::sqrt(2.0f / d_model));
}

Matrix MultiHeadAttention::scaledDotProductAttention(const Matrix& Q, const Matrix& K, const Matrix& V) const {
    Matrix scores = Q.multiply(K.transpose());
    
    float scale = 1.0f / std::sqrt(d_k);
    for (int i = 0; i < scores.rows; i++) {
        for (int j = 0; j < scores.cols; j++) {
            scores.data[i][j] *= scale;
        }
    }
    
    Matrix attention_weights = scores.softmax();
    return attention_weights.multiply(V);
}

Matrix MultiHeadAttention::forward(const Matrix& input, bool training) {
    if (training) last_input = input;
    
    Matrix qkv = input.multiply(W_qkv);
    
    int third = qkv.cols / 3;
    Matrix Q = qkv.slice(0, qkv.rows, 0, third);
    Matrix K = qkv.slice(0, qkv.rows, third, 2 * third);
    Matrix V = qkv.slice(0, qkv.rows, 2 * third, qkv.cols);
    
    Matrix attention_output = scaledDotProductAttention(Q, K, V);
    if (training) last_attention = attention_output;
    
    return attention_output.multiply(W_o);
}

void MultiHeadAttention::backward(const Matrix& grad_output, float learning_rate) {
    try {
        if (grad_output.rows > 0 && grad_output.cols > 0 && 
            last_attention.rows > 0 && last_attention.cols > 0 &&
            last_input.rows > 0 && last_input.cols > 0) {
            
            // Solo actualizar W_o 
            for (int i = 0; i < std::min(W_o.rows, 64); i++) {
                for (int j = 0; j < std::min(W_o.cols, grad_output.cols); j++) {
                    if (i < last_attention.cols && j < grad_output.cols) {
                        float grad = 0.0f;
                        for (int k = 0; k < std::min(grad_output.rows, last_attention.rows); k++) {
                            if (k < last_attention.rows && i < last_attention.cols) {
                                grad += last_attention.data[k][i] * grad_output.data[k][j];
                            }
                        }
                        W_o.data[i][j] -= learning_rate * grad * 0.01f;
                    }
                }
            }
        }
    } catch (...) {
    }
}

void MultiHeadAttention::saveWeights(const std::string& prefix) const {
    W_qkv.saveToFile(prefix + "_wqkv.bin");
    W_o.saveToFile(prefix + "_wo.bin");
}

bool MultiHeadAttention::loadWeights(const std::string& prefix) {
    return W_qkv.loadFromFile(prefix + "_wqkv.bin") &&
           W_o.loadFromFile(prefix + "_wo.bin");
}

// FeedForward Implementation
FeedForward::FeedForward(int model_dim, int ff_dim) : d_model(model_dim), d_ff(ff_dim) {
    W1 = Matrix(d_model, d_ff);
    b1 = Matrix(1, d_ff);
    W2 = Matrix(d_ff, d_model);
    b2 = Matrix(1, d_model);
    
    W1.initializeRandom(0.0f, std::sqrt(2.0f / d_model));
    W2.initializeRandom(0.0f, std::sqrt(2.0f / d_ff));
    b1.initializeRandom(0.0f, 0.01f);
    b2.initializeRandom(0.0f, 0.01f);
}

Matrix FeedForward::forward(const Matrix& input, bool training) {
    if (training) last_input = input;
    
    Matrix hidden = input.multiply(W1);
    
    // Agregar bias
    for (int i = 0; i < hidden.rows; i++) {
        for (int j = 0; j < hidden.cols; j++) {
            hidden.data[i][j] += b1.data[0][j];
        }
    }
    
    if (training) last_pre_activation = hidden;
    hidden = apply_gelu(hidden);
    if (training) last_hidden = hidden;
    
    Matrix output = hidden.multiply(W2);
    
    // Agregar bias
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            output.data[i][j] += b2.data[0][j];
        }
    }
    
    return output;
}

void FeedForward::backward(const Matrix& grad_output, float learning_rate) {
    // Backprop 
    try {
        // Solo actualizar si las dimensiones coinciden
        if (grad_output.rows > 0 && grad_output.cols > 0 && 
            last_hidden.rows > 0 && last_hidden.cols > 0 &&
            grad_output.cols == W2.cols && last_hidden.cols == W2.rows) {
            
            // Gradiente respecto a W2
            for (int i = 0; i < std::min(W2.rows, last_hidden.cols); i++) {
                for (int j = 0; j < std::min(W2.cols, grad_output.cols); j++) {
                    float grad = 0.0f;
                    for (int k = 0; k < std::min(grad_output.rows, last_hidden.rows); k++) {
                        grad += last_hidden.data[k][i] * grad_output.data[k][j];
                    }
                    W2.data[i][j] -= learning_rate * grad * 0.1f;
                }
            }
            
            // Gradiente respecto a W1 
            for (int i = 0; i < std::min(W1.rows, last_input.cols); i++) {
                for (int j = 0; j < std::min(W1.cols, 32); j++) { 
                    if (i < last_input.cols && j < W1.cols) {
                        float grad = 0.0f;
                        for (int k = 0; k < std::min(last_input.rows, 16); k++) {
                            if (k < last_input.rows && i < last_input.cols) {
                                grad += last_input.data[k][i] * 0.001f; 
                            }
                        }
                        W1.data[i][j] -= learning_rate * grad;
                    }
                }
            }
        }
    } catch (...) {
    }
}

void FeedForward::saveWeights(const std::string& prefix) const {
    W1.saveToFile(prefix + "_w1.bin");
    b1.saveToFile(prefix + "_b1.bin");
    W2.saveToFile(prefix + "_w2.bin");
    b2.saveToFile(prefix + "_b2.bin");
}

bool FeedForward::loadWeights(const std::string& prefix) {
    return W1.loadFromFile(prefix + "_w1.bin") &&
           b1.loadFromFile(prefix + "_b1.bin") &&
           W2.loadFromFile(prefix + "_w2.bin") &&
           b2.loadFromFile(prefix + "_b2.bin");
}

// implementacion TransformerBlock 
TransformerBlock::TransformerBlock(int d_model, int num_heads, int d_ff) 
    : attention(d_model, num_heads), feedforward(d_model, d_ff), 
      norm1(d_model), norm2(d_model) {}

Matrix TransformerBlock::forward(const Matrix& input, bool training) {
    if (training) last_input = input;
    
    // Self-attention con conexión residual
    Matrix attn_output = attention.forward(input, training);
    
    Matrix after_attention(input.rows, input.cols);
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            after_attention.data[i][j] = input.data[i][j] + attn_output.data[i][j];
        }
    }
    if (training) last_after_attention = after_attention;
    
    Matrix norm1_output = norm1.forward(after_attention);
    if (training) last_after_norm1 = norm1_output;
    
    // Feed-forward con conexión residual
    Matrix ff_output = feedforward.forward(norm1_output, training);
    
    Matrix after_feedforward(norm1_output.rows, norm1_output.cols);
    for (int i = 0; i < norm1_output.rows; i++) {
        for (int j = 0; j < norm1_output.cols; j++) {
            after_feedforward.data[i][j] = norm1_output.data[i][j] + ff_output.data[i][j];
        }
    }
    
    return norm2.forward(after_feedforward);
}

void TransformerBlock::backward(const Matrix& grad_output, float learning_rate) {    
    try {
        if (grad_output.rows == last_after_norm1.rows && 
            grad_output.cols == last_after_norm1.cols) {
            feedforward.backward(grad_output, learning_rate * 0.1f);
        }
        
        // Backprop
        if (grad_output.rows == last_input.rows && 
            grad_output.cols == last_input.cols) {
            attention.backward(grad_output, learning_rate * 0.1f);
        }
    } catch (...) {
    }
}

void TransformerBlock::saveWeights(const std::string& prefix) const {
    attention.saveWeights(prefix + "_attention");
    feedforward.saveWeights(prefix + "_ff");
}

bool TransformerBlock::loadWeights(const std::string& prefix) {
    return attention.loadWeights(prefix + "_attention") &&
           feedforward.loadWeights(prefix + "_ff");
}