#include "trainer.h"
#include "transformer.h"
#include "matrix.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<int>> 
MNISTLoader::loadMNISTCSV(const std::string& filename, int max_samples) {
    std::vector<std::vector<std::vector<float>>> images;
    std::vector<int> labels;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Error: No se pudo abrir el archivo: " << filename << std::endl;
        return {images, labels};
    }
    
    std::cout << "Cargando dataset desde: " << filename << std::endl;
    
    std::string line;
    int line_count = 0;
    int loaded_count = 0;
    
    while (std::getline(file, line) && (max_samples == -1 || loaded_count < max_samples)) {
        line_count++;
        
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        
        if (line_count % 1000 == 0) {
            std::cout << "Procesando linea " << line_count << "..." << std::endl;
        }
        
        try {
            std::vector<int> values;
            std::stringstream ss(line);
            std::string token;
            
            while (std::getline(ss, token, ',')) {
                token.erase(0, token.find_first_not_of(" \t"));
                token.erase(token.find_last_not_of(" \t") + 1);
                
                if (!token.empty()) {
                    values.push_back(std::stoi(token));
                }
            }
            
            if (values.size() == 785) {
                int label = values[0];
                std::vector<std::vector<float>> image(28, std::vector<float>(28));
                
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        int pixel_idx = i * 28 + j + 1;
                        image[i][j] = values[pixel_idx] / 255.0f;
                    }
                }
                
                images.push_back(image);
                labels.push_back(label);
                loaded_count++;
                
                if (loaded_count % 500 == 0) {
                    std::cout << "Cargadas " << loaded_count << " imagenes..." << std::endl;
                }
            }
            
        } catch (const std::exception& e) {
            if (line_count <= 10) {
                std::cout << "Error en linea " << line_count << ": " << e.what() << std::endl;
            }
        }
    }
    
    file.close();
    std::cout << "Dataset cargado: " << images.size() << " imagenes" << std::endl;
    return {images, labels};
}

float evaluateModel(VisionTransformer& model, 
                   const std::vector<std::vector<std::vector<float>>>& images,
                   const std::vector<int>& labels) {
    int correct = 0;
    int total = images.size();
    
    for (int i = 0; i < total; i++) {
        Matrix output = model.forward(images[i], false);
        Matrix probs = output.softmax();
        
        int predicted = 0;
        float max_prob = probs.data[0][0];
        for (int j = 1; j < probs.cols; j++) {
            if (probs.data[0][j] > max_prob) {
                max_prob = probs.data[0][j];
                predicted = j;
            }
        }
        
        if (predicted == labels[i]) {
            correct++;
        }
    }
    
    return (float)correct / total * 100.0f;
}