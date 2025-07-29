#ifndef LAYERS_H
#define LAYERS_H

#include "matrix.h"
#include <vector>
#include <string>

// Clase para manejar embeddings posicionales
class PositionalEncoding {
private:
    Matrix encoding;
    int max_len, d_model;
    
public:
    PositionalEncoding(int max_length, int model_dim);
    Matrix getEncoding(int seq_len) const;
};

// Capa de normalizacion
class LayerNorm {
private:
    std::vector<float> gamma, beta;
    int d_model;
    float eps;
    
public:
    LayerNorm(int model_dim, float epsilon = 1e-6);
    Matrix forward(const Matrix& input) const;
    void saveWeights(const std::string& prefix) const;
    bool loadWeights(const std::string& prefix);
};

// Mecanismo de atencion multi-cabeza 
class MultiHeadAttention {
private:
    Matrix W_qkv, W_o;
    int d_model, num_heads, d_k;
    Matrix last_input, last_attention;  // Para backprop
    
public:
    MultiHeadAttention(int model_dim, int heads);
    Matrix scaledDotProductAttention(const Matrix& Q, const Matrix& K, const Matrix& V) const;
    Matrix forward(const Matrix& input, bool training = false);
    void backward(const Matrix& grad_output, float learning_rate);
    void saveWeights(const std::string& prefix) const;
    bool loadWeights(const std::string& prefix);
};

// Feed Forward Network
class FeedForward {
public: 
    Matrix W1, b1, W2, b2;
    Matrix last_input, last_hidden, last_pre_activation;
    int d_model, d_ff;
    
public:
    FeedForward(int model_dim, int ff_dim);
    Matrix forward(const Matrix& input, bool training = false);
    void backward(const Matrix& grad_output, float learning_rate);
    void saveWeights(const std::string& prefix) const;
    bool loadWeights(const std::string& prefix);
};

// Bloque Transformer
class TransformerBlock {
private:
    MultiHeadAttention attention;
    FeedForward feedforward;
    LayerNorm norm1, norm2;
    Matrix last_input, last_after_attention, last_after_norm1;  // Para backprop
    
public:
    TransformerBlock(int d_model, int num_heads, int d_ff);
    Matrix forward(const Matrix& input, bool training = false);
    void backward(const Matrix& grad_output, float learning_rate);
    void saveWeights(const std::string& prefix) const;
    bool loadWeights(const std::string& prefix);
};

#endif // LAYERS_H