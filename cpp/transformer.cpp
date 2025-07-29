#include "transformer.h"
#include "activations.h"
#include "matrix.h"
#include "layers.h"
#include <iostream>
#include <cassert>
#include <random>
#include <cmath>
#include <fstream>
VisionTransformer::VisionTransformer(int img_sz, int patch_sz, int model_dim, int num_heads, 
                                   int num_layers, int d_ff, int num_cls, bool adam_enabled) 
    : img_size(img_sz), patch_size(patch_sz), d_model(model_dim), num_classes(num_cls),
      pos_encoding(1000, model_dim), final_norm(model_dim), use_adam(adam_enabled),
      optimizer(0.001f) {
    
    adjustPatchSize();
    
    num_patches = (img_size / patch_size) * (img_size / patch_size);
    int patch_dim = patch_size * patch_size;
    
    std::cout << "Inicializando Vision Transformer:" << std::endl;
    std::cout << "  - Imagen: " << img_size << "x" << img_size << std::endl;
    std::cout << "  - Patch: " << patch_size << "x" << patch_size << std::endl;
    std::cout << "  - Num patches: " << num_patches << std::endl;
    std::cout << "  - Patch dim: " << patch_dim << std::endl;
    std::cout << "  - Model dim: " << d_model << std::endl;
    if (use_adam) {
        int num_weights = 3; 
        int num_biases = 0;  
        optimizer.initialize(num_weights, num_biases);
        std::cout << "Adam optimizer " << std::endl;
    } else {
        std::cout << "Usando SGD" << std::endl;
    }
    
    // Inicializar componentes
    patch_embedding = Matrix(patch_dim, d_model);
    patch_embedding.initializeRandom(0.0f, std::sqrt(2.0f / patch_dim));
    
    class_token = Matrix(1, d_model);
    class_token.initializeRandom(0.0f, 0.02f);
    
    // Crear bloques transformer
    for (int i = 0; i < num_layers; i++) {
        transformer_blocks.emplace_back(d_model, num_heads, d_ff);
    }
    
    classifier_head = Matrix(d_model, num_classes);
    classifier_head.initializeRandom(0.0f, std::sqrt(2.0f / d_model));
    
    std::cout << "Vision Transformer inicializado con " << getParameterCount() << " parametros" << std::endl;
}

void VisionTransformer::adjustPatchSize() {
    if (img_size % patch_size != 0) {
        std::cout << "ADVERTENCIA: " << img_size << " no es divisible por " << patch_size << std::endl;
        std::cout << "Ajustando patch_size a divisor valido..." << std::endl;
        
        for (int p = patch_size; p >= 1; p--) {
            if (img_size % p == 0) {
                patch_size = p;
                break;
            }
        }
        std::cout << "Nuevo patch_size: " << patch_size << std::endl;
    }
}

Matrix VisionTransformer::patchify(const std::vector<std::vector<float>>& image) const {
    validateImageSize(image);
    
    Matrix patches(num_patches, patch_size * patch_size);
    
    int patch_idx = 0;
    for (int i = 0; i < img_size; i += patch_size) {
        for (int j = 0; j < img_size; j += patch_size) {
            if (patch_idx >= num_patches) break;
            
            int pixel_idx = 0;
            for (int pi = 0; pi < patch_size; pi++) {
                for (int pj = 0; pj < patch_size; pj++) {
                    if (i + pi < img_size && j + pj < img_size) {
                        patches.data[patch_idx][pixel_idx] = image[i + pi][j + pj];
                    }
                    pixel_idx++;
                }
            }
            patch_idx++;
        }
    }
    return patches;
}

void VisionTransformer::validateImageSize(const std::vector<std::vector<float>>& image) const {
    assert(image.size() == img_size);
    for (const auto& row : image) {
        assert(row.size() == img_size);
    }
}

Matrix VisionTransformer::forward(const std::vector<std::vector<float>>& image, bool training) {
    Matrix patches = patchify(image);
    if (training) last_patches = patches;
    
    Matrix patch_embeddings = patches.multiply(patch_embedding);
    if (training) last_patch_embeddings = patch_embeddings;
    
    // Agregar class token
    Matrix input_embeddings(num_patches + 1, d_model);
    for (int j = 0; j < d_model; j++) {
        input_embeddings.data[0][j] = class_token.data[0][j];
    }
    for (int i = 0; i < num_patches; i++) {
        for (int j = 0; j < d_model; j++) {
            input_embeddings.data[i + 1][j] = patch_embeddings.data[i][j];
        }
    }
    
    // Agregar encoding posicional
    Matrix pos_enc = pos_encoding.getEncoding(num_patches + 1);
    for (int i = 0; i < input_embeddings.rows; i++) {
        for (int j = 0; j < input_embeddings.cols; j++) {
            input_embeddings.data[i][j] += pos_enc.data[i][j];
        }
    }
    
    if (training) last_input_embeddings = input_embeddings;
    
    // Pasar por bloques transformer
    Matrix output = input_embeddings;
    for (auto& block : transformer_blocks) {
        output = block.forward(output, training);
    }
    
    output = final_norm.forward(output);
    
    // Clasificación usando solo el class token
    Matrix class_output(1, d_model);
    for (int j = 0; j < d_model; j++) {
        class_output.data[0][j] = output.data[0][j];
    }
    
    if (training) last_class_output = class_output;
    
    return class_output.multiply(classifier_head);
}

void VisionTransformer::train(const std::vector<std::vector<float>>& image, int true_label, float learning_rate) {
    Matrix output = forward(image, true);
    Matrix probs = output.softmax();
    
    // Calcular gradiente de cross-entropy + softmax
    Matrix grad_output(1, num_classes);
    for (int i = 0; i < num_classes; i++) {
        grad_output.data[0][i] = probs.data[0][i];
        if (i == true_label) {
            grad_output.data[0][i] -= 1.0f;
        }
    }
    if (use_adam) {
        //  ADAM OPTIMIZER
        optimizer.setLearningRate(learning_rate);
        trainWithAdam(grad_output);
        optimizer.step();
    } else {
    try {
        // Actualizar classifier head
        backpropClassifier(grad_output, learning_rate);
        
        // Gradiente hacia class token
        Matrix grad_class_token(1, d_model);
        for (int j = 0; j < d_model && j < grad_output.cols; j++) {
            float grad_sum = 0.0f;
            for (int i = 0; i < num_classes && i < classifier_head.rows; i++) {
                if (j < classifier_head.cols && i < grad_output.cols) {
                    grad_sum += classifier_head.data[j][i] * grad_output.data[0][i];
                }
            }
            grad_class_token.data[0][j] = grad_sum;
        }
        
        //  Backprop  a través de transformer blocks
        for (auto it = transformer_blocks.rbegin(); it != transformer_blocks.rend(); ++it) {
            it->backward(grad_class_token, learning_rate * 0.01f); // LR muy bajo
        }
        
        //  Actualizar patch embedding y class token
        backpropPatchEmbedding(grad_class_token, learning_rate);
        backpropClassToken(grad_class_token, learning_rate);
        
    } catch (...) {
        try {
            backpropClassifier(grad_output, learning_rate);
            
            Matrix grad_class_token(1, d_model);
            for (int j = 0; j < d_model; j++) {
                grad_class_token.data[0][j] = 0.001f; 
            }
            
            backpropClassToken(grad_class_token, learning_rate);
        } catch (...) {
        }
    }
}}

void VisionTransformer::backpropClassifier(const Matrix& grad_output, float learning_rate) {
    // Dropout para regularización
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dropout_dist(0.0f, 1.0f);
    
    if (last_class_output.rows == 0 || last_class_output.cols == 0 ||
        grad_output.rows == 0 || grad_output.cols == 0) {
        return;
    }
    
    for (int i = 0; i < classifier_head.rows && i < last_class_output.cols; i++) {
        for (int j = 0; j < classifier_head.cols && j < grad_output.cols; j++) {
            if (dropout_dist(gen) > 0.2f) { 
                
                // Gradiente
                float grad = 0.0f;
                for (int k = 0; k < std::min(last_class_output.rows, grad_output.rows); k++) {
                    if (i < last_class_output.cols && j < grad_output.cols) {
                        grad += last_class_output.data[k][i] * grad_output.data[k][j];
                    }
                }
                
                // L2 regularization
                float l2_penalty = 0.001f * classifier_head.data[i][j];
                classifier_head.data[i][j] -= learning_rate * (grad + l2_penalty);
            }
        }
    }
}

void VisionTransformer::backpropPatchEmbedding(const Matrix& grad_class_token, float learning_rate) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dropout_dist(0.0f, 1.0f);
    
    if (last_patches.rows == 0 || last_patches.cols == 0 || 
        grad_class_token.rows == 0 || grad_class_token.cols == 0) {
        return;
    }

    for (int i = 0; i < patch_embedding.rows && i < last_patches.cols; i++) {
        for (int j = 0; j < patch_embedding.cols && j < grad_class_token.cols; j++) {
            if (dropout_dist(gen) > 0.3f) {
                
                // Calcular gradiente promedio de todos los patches
                float grad_sum = 0.0f;
                int count = 0;
                
                for (int p = 0; p < last_patches.rows && p < num_patches; p++) {
                    if (i < last_patches.cols) {
                        grad_sum += last_patches.data[p][i] * grad_class_token.data[0][j];
                        count++;
                    }
                }
                
                if (count > 0) {
                    float avg_grad = grad_sum / count;
                    float l2_penalty = 0.0005f * patch_embedding.data[i][j];
                    patch_embedding.data[i][j] -= learning_rate * 0.01f * (avg_grad + l2_penalty);
                }
            }
        }
    }
}

void VisionTransformer::backpropClassToken(const Matrix& grad_class_token, float learning_rate) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dropout_dist(0.0f, 1.0f);
    
    for (int j = 0; j < d_model && j < grad_class_token.cols; j++) {
        if (dropout_dist(gen) > 0.25f) { 
            float l2_penalty = 0.0005f * class_token.data[0][j];
            class_token.data[0][j] -= learning_rate * 0.05f * (grad_class_token.data[0][j] + l2_penalty);
        }
    }
}

int VisionTransformer::predict(const std::vector<std::vector<float>>& image) {
    Matrix output = forward(image, false);
    Matrix probs = output.softmax();
    
    int predicted = 0;
    float max_prob = probs.data[0][0];
    for (int j = 1; j < probs.cols; j++) {
        if (probs.data[0][j] > max_prob) {
            max_prob = probs.data[0][j];
            predicted = j;
        }
    }
    
    return predicted;
}

std::vector<float> VisionTransformer::getProbabilities(const std::vector<std::vector<float>>& image) {
    Matrix output = forward(image, false);
    Matrix probs = output.softmax();
    
    std::vector<float> probabilities;
    for (int j = 0; j < probs.cols; j++) {
        probabilities.push_back(probs.data[0][j]);
    }
    
    return probabilities;
}
void VisionTransformer::saveModel(const std::string& model_name) const {
    std::cout << "Guardando modelo: " << model_name << std::endl;
    
    std::string filename = model_name + "_pesos.bin";
    std::ofstream file(filename, std::ios::binary);
    
    if (!file) {
        std::cout << "Error: No se pudo crear el archivo " << filename << std::endl;
        return;
    }
    
    // Guardar patch_embedding
    file.write(reinterpret_cast<const char*>(&patch_embedding.rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&patch_embedding.cols), sizeof(int));
    for (int i = 0; i < patch_embedding.rows; i++) {
        for (int j = 0; j < patch_embedding.cols; j++) {
            file.write(reinterpret_cast<const char*>(&patch_embedding.data[i][j]), sizeof(float));
        }
    }
    
    //  Guardar class_token
    file.write(reinterpret_cast<const char*>(&class_token.rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&class_token.cols), sizeof(int));
    for (int i = 0; i < class_token.rows; i++) {
        for (int j = 0; j < class_token.cols; j++) {
            file.write(reinterpret_cast<const char*>(&class_token.data[i][j]), sizeof(float));
        }
    }
    
    //  Guardar classifier_head
    file.write(reinterpret_cast<const char*>(&classifier_head.rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&classifier_head.cols), sizeof(int));
    for (int i = 0; i < classifier_head.rows; i++) {
        for (int j = 0; j < classifier_head.cols; j++) {
            file.write(reinterpret_cast<const char*>(&classifier_head.data[i][j]), sizeof(float));
        }
    }
    
    file.close();
    std::cout << "Modelo guardado en: " << filename << std::endl;
}
bool VisionTransformer::loadModel(const std::string& model_name) {
    std::cout << "Cargando modelo: " << model_name << std::endl;
    
    std::string filename = model_name + "_pesos.bin";
    std::ifstream file(filename, std::ios::binary);
    
    if (!file) {
        std::cout << "Error: No se encontro el archivo " << filename << std::endl;
        return false;
    }
    
    try {
        //  Cargar patch_embedding
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        
        if (rows != patch_embedding.rows || cols != patch_embedding.cols) {
            std::cout << "Error: Dimensiones de patch_embedding no coinciden" << std::endl;
            return false;
        }
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                file.read(reinterpret_cast<char*>(&patch_embedding.data[i][j]), sizeof(float));
            }
        }
        
        // Cargar class_token
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        
        if (rows != class_token.rows || cols != class_token.cols) {
            std::cout << "Error: Dimensiones de class_token no coinciden" << std::endl;
            return false;
        }
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                file.read(reinterpret_cast<char*>(&class_token.data[i][j]), sizeof(float));
            }
        }
        
        // Cargar classifier_head
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        
        if (rows != classifier_head.rows || cols != classifier_head.cols) {
            std::cout << "Error: Dimensiones de classifier_head no coinciden" << std::endl;
            return false;
        }
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                file.read(reinterpret_cast<char*>(&classifier_head.data[i][j]), sizeof(float));
            }
        }
        
        file.close();
        std::cout << "Modelo cargado exitosamente desde: " << filename << std::endl;
        return true;
        
    } catch (...) {
        std::cout << "Error: Fallo al cargar el modelo" << std::endl;
        file.close();
        return false;
    }
}

void VisionTransformer::printModelInfo() const {
    std::cout << "\n=== Informacion del Modelo ===" << std::endl;
    std::cout << "Parametros del modelo:" << std::endl;
    std::cout << "  - Tamaño de imagen: " << img_size << "x" << img_size << std::endl;
    std::cout << "  - Tamaño de patch: " << patch_size << "x" << patch_size << std::endl;
    std::cout << "  - Numero de patches: " << num_patches << std::endl;
    std::cout << "  - Dimension del modelo: " << d_model << std::endl;
    std::cout << "  - Numero de clases: " << num_classes << std::endl;
    std::cout << "  - Numero de bloques: " << transformer_blocks.size() << std::endl;
    std::cout << "  - Total de parametros: " << getParameterCount() << std::endl;
}

int VisionTransformer::getParameterCount() const {
    int total = 0;
    
    // Patch embedding
    total += patch_embedding.rows * patch_embedding.cols;
    
    // Class token
    total += class_token.rows * class_token.cols;
    
    // Classifier head
    total += classifier_head.rows * classifier_head.cols;
    
    // Transformer blocks 
    for (const auto& block : transformer_blocks) {
        total += d_model * d_model * 4;
        total += d_model * 512 * 2; 
    }
    
    return total;
}
void VisionTransformer::trainWithAdam(const Matrix& grad_output) {
    int param_index = 0;
    
    try {
        // ACTUALIZAR CLASSIFIER HEAD
        if (last_class_output.rows > 0 && last_class_output.cols > 0) {
            Matrix grad_classifier(d_model, num_classes);
            
            for (int i = 0; i < d_model && i < last_class_output.cols; i++) {
                for (int j = 0; j < num_classes && j < grad_output.cols; j++) {
                    grad_classifier.data[i][j] = last_class_output.data[0][i] * grad_output.data[0][j];
                }
            }
            
            optimizer.updateWeights(classifier_head, grad_classifier, param_index++);
        }
        
        // CALCULAR GRADIENTE HACIA CLASS TOKEN
        Matrix grad_class_token(1, d_model);
        for (int j = 0; j < d_model && j < grad_output.cols; j++) {
            float grad_sum = 0.0f;
            for (int i = 0; i < num_classes && i < classifier_head.rows; i++) {
                if (j < classifier_head.cols && i < grad_output.cols) {
                    grad_sum += classifier_head.data[j][i] * grad_output.data[0][i];
                }
            }
            grad_class_token.data[0][j] = grad_sum;
        }
        
        // ACTUALIZAR CLASS TOKEN
        Matrix class_grad(1, d_model);
        for (int j = 0; j < d_model; j++) {
            class_grad.data[0][j] = grad_class_token.data[0][j] * 0.1f;
        }
        optimizer.updateWeights(class_token, class_grad, param_index++);
        
        //  ACTUALIZAR PATCH EMBEDDING
        if (last_patches.rows > 0 && last_patches.cols > 0) {
            Matrix patch_grad(patch_embedding.rows, patch_embedding.cols);
            
            for (int i = 0; i < patch_embedding.rows && i < last_patches.cols; i++) {
                for (int j = 0; j < patch_embedding.cols && j < grad_class_token.cols; j++) {
                    float grad_sum = 0.0f;
                    int count = 0;
                    
                    for (int p = 0; p < last_patches.rows && p < num_patches; p++) {
                        grad_sum += last_patches.data[p][i] * grad_class_token.data[0][j];
                        count++;
                    }
                    
                    patch_grad.data[i][j] = (count > 0) ? (grad_sum / count * 0.01f) : 0.0f;
                }
            }
            
            optimizer.updateWeights(patch_embedding, patch_grad, param_index++);
        }
        
    } catch (const std::exception& e) {
        backpropClassifier(grad_output, optimizer.getLearningRate());
    }
}