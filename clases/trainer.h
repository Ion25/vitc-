#ifndef TRAINER_H
#define TRAINER_H

#include <vector>
#include <string>
#include <utility>

// Clase para cargar datasets MNIST
class MNISTLoader {
public:
    static std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<int>> 
    loadMNISTCSV(const std::string& filename, int max_samples = -1);
};

// Función para evaluar el modelo
class VisionTransformer; // Forward declaration

float evaluateModel(VisionTransformer& model, 
                   const std::vector<std::vector<std::vector<float>>>& images,
                   const std::vector<int>& labels);

#endif // TRAINER_H