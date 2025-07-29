#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "matrix.h"
#include "layers.h"
#include "adam_optimizer.h"
#include <vector>
#include <string>

class VisionTransformer {
private:
    // componentes
    Matrix patch_embedding;
    Matrix class_token;
    PositionalEncoding pos_encoding;
    std::vector<TransformerBlock> transformer_blocks;
    Matrix classifier_head;
    LayerNorm final_norm;
    
    // Parametros del modelo
    int patch_size, num_patches, d_model, num_classes, img_size;
    
    // Para backpropagation
    Matrix last_class_output;
    Matrix last_patches;
    Matrix last_patch_embeddings;
    Matrix last_input_embeddings;
    AdamOptimizer optimizer;
    bool use_adam;
    void trainWithAdam(const Matrix& grad_output);
public:
    // constructor
    VisionTransformer(int img_size, int patch_sz, int model_dim, int num_heads,
                  int num_layers, int d_ff, int num_cls, bool adam_enabled = false);
    
    // Forward pass
    Matrix forward(const std::vector<std::vector<float>>& image, bool training = false);
    
    // Entrenamiento con backpropagation
    void train(const std::vector<std::vector<float>>& image, int true_label, float learning_rate);
    
    // Prediccion
    int predict(const std::vector<std::vector<float>>& image);
    std::vector<float> getProbabilities(const std::vector<std::vector<float>>& image);
    
    // Persistencia del modelo
    void saveModel(const std::string& model_name) const;
    bool loadModel(const std::string& model_name);
    
    // Utilidades
    void printModelInfo() const;
    int getParameterCount() const;
    
private:
    // Funciones auxiliares
    Matrix patchify(const std::vector<std::vector<float>>& image) const;
    void validateImageSize(const std::vector<std::vector<float>>& image) const;
    void adjustPatchSize();
    
    // Backpropagation
    void backpropClassifier(const Matrix& grad_output, float learning_rate);
    void backpropPatchEmbedding(const Matrix& grad_class_token, float learning_rate);
    void backpropClassToken(const Matrix& grad_class_token, float learning_rate);
};

#endif