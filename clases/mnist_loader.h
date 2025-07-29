#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>
#include <utility>

class MNISTLoader {
public:
    // Cargar datos desde CSV
    static std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<int>> 
    loadMNISTCSV(const std::string& filename, int max_samples = -1);
    
    static std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<int>> 
    createSyntheticData(int num_samples = 100);
    
    // Data augmentation
    static std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<int>>
    augmentData(const std::vector<std::vector<std::vector<float>>>& images,
                const std::vector<int>& labels,
                float noise_level = 0.02f);
    
    // Utilidades
    static void showImageStats(const std::vector<std::vector<float>>& image, int label);
    static bool validateDataset(const std::vector<std::vector<std::vector<float>>>& images, 
                               const std::vector<int>& labels);
    
    // Conversión de imagen 2D a vector 1D
    static std::vector<float> flattenImage(const std::vector<std::vector<float>>& image);
    
    // Normalización de datos
    static void normalizeImages(std::vector<std::vector<std::vector<float>>>& images);
};

#endif