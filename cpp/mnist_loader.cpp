#include "mnist_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cctype>

std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<int>>
MNISTLoader::loadMNISTCSV(const std::string& filename, int max_samples) {
    std::vector<std::vector<std::vector<float>>> images;
    std::vector<int> labels;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error al abrir el archivo: " << filename << std::endl;
        return {images, labels};
    }

    std::string first_line;
    std::getline(file, first_line);
    bool has_header = false;
    for (char c : first_line) {
        if (c == ',' || isspace(c)) continue;
        if (!isdigit(c)) {
            has_header = true;
            break;
        }
    }
    if (!has_header) {
        file.seekg(0);
    }

    std::string line;
    int sample_count = 0;

    while (std::getline(file, line)) {
        if (max_samples != -1 && sample_count >= max_samples) break;

        std::stringstream ss(line);
        std::string value;
        std::vector<float> pixels;
        int label = -1;

        // Leer etiqueta
        if (!std::getline(ss, value, ',')) continue;

        try {
            label = std::stoi(value);
            if (label < 0 || label > 9) {
                std::cerr << "Etiqueta invalida (" << label << ") en linea " << sample_count + 1 << std::endl;
                continue;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error en linea " << sample_count + 1 << ": " << e.what() << " -> '" << value << "'" << std::endl;
            continue;
        }

        // Leer pixeles
        bool pixel_error = false;
        while (std::getline(ss, value, ',')) {
            try {
                float pixel_val = std::stof(value);
                if (pixel_val < 0 || pixel_val > 255) {
                    std::cerr << "Advertencia: Pixel fuera de rango (" << pixel_val << ") en linea " << sample_count + 1 << std::endl;
                    pixel_val = std::max(0.0f, std::min(255.0f, pixel_val));
                }
                pixels.push_back(pixel_val / 255.0f);
            } catch (...) {
                pixels.push_back(0.0f);
                pixel_error = true;
            }
        }

        if (pixel_error) {
            std::cerr << "Errores de conversion en pixeles (linea " << sample_count + 1 << ")" << std::endl;
        }

        if (pixels.size() != 28 * 28) {
            std::cerr << "Imagen corrupta (tamaño incorrecto): " << pixels.size() 
                      << " valores. Se esperaban " << 28*28 << std::endl;
            continue;
        }

        // Convertir a imagen 28x28
        std::vector<std::vector<float>> image(28, std::vector<float>(28));
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                image[i][j] = pixels[i * 28 + j];
            }
        }

        images.push_back(image);
        labels.push_back(label);
        ++sample_count;
    }

    std::cout << "Cargadas " << images.size() << " imagenes de " << filename << std::endl;
    return {images, labels};
}


std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<int>> 
MNISTLoader::createSyntheticData(int num_samples) {
    std::vector<std::vector<std::vector<float>>> images;
    std::vector<int> labels;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> label_dist(0, 9);
        
    for (int i = 0; i < num_samples; i++) {
        std::vector<std::vector<float>> image(28, std::vector<float>(28));
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                image[r][c] = dist(gen);
            }
        }
        
        images.push_back(image);
        labels.push_back(label_dist(gen));
    }
    
    return {images, labels};
}

std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<int>>
MNISTLoader::augmentData(const std::vector<std::vector<std::vector<float>>>& images,
                        const std::vector<int>& labels,
                        float noise_level) {
    std::vector<std::vector<std::vector<float>>> augmented_images = images;
    std::vector<int> augmented_labels = labels;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, noise_level);
    
    std::cout << "Aplicando data augmentation con ruido " << noise_level << "..." << std::endl;
    
    for (size_t i = 0; i < images.size(); i++) {
        std::vector<std::vector<float>> noisy_image = images[i];
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                noisy_image[r][c] += noise(gen);
                noisy_image[r][c] = std::max(0.0f, std::min(1.0f, noisy_image[r][c]));
            }
        }
        augmented_images.push_back(noisy_image);
        augmented_labels.push_back(labels[i]);
    }
    
    return {augmented_images, augmented_labels};
}

void MNISTLoader::showImageStats(const std::vector<std::vector<float>>& image, int label) {
    float min_val = 1.0f, max_val = 0.0f, sum = 0.0f;
    int total_pixels = 0;
    
    for (const auto& row : image) {
        for (float pixel : row) {
            min_val = std::min(min_val, pixel);
            max_val = std::max(max_val, pixel);
            sum += pixel;
            total_pixels++;
        }
    }
    
    float mean = sum / total_pixels;
    
    std::cout << "Estadisticas de imagen (etiqueta " << label << "):" << std::endl;
    std::cout << "   - Rango de pixeles: [" << std::fixed << std::setprecision(3) 
              << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "   - Valor promedio: " << mean << std::endl;
}

bool MNISTLoader::validateDataset(const std::vector<std::vector<std::vector<float>>>& images, 
                                  const std::vector<int>& labels) {
    if (images.size() != labels.size()) {
        std::cout << "Error: Numero de imagenes (" << images.size() 
                  << ") no coincide con numero de etiquetas (" << labels.size() << ")" << std::endl;
        return false;
    }
    
    if (images.empty()) {
        std::cout << "Error: Dataset vacio" << std::endl;
        return false;
    }
    
    // Verificar que todas las imágenes tienen el tamaño correcto
    for (size_t i = 0; i < images.size(); i++) {
        if (images[i].size() != 28) {
            std::cout << "Error: Imagen " << i << " tiene " << images[i].size() 
                      << " filas (esperaba 28)" << std::endl;
            return false;
        }
        
        for (size_t j = 0; j < images[i].size(); j++) {
            if (images[i][j].size() != 28) {
                std::cout << "Error: Imagen " << i << ", fila " << j << " tiene " 
                          << images[i][j].size() << " columnas (esperaba 28)" << std::endl;
                return false;
            }
        }
    }
    
    // Verificar rango de etiquetas
    int min_label = *std::min_element(labels.begin(), labels.end());
    int max_label = *std::max_element(labels.begin(), labels.end());
    
    if (min_label < 0 || max_label > 9) {
        std::cout << "Advertencia: Etiquetas fuera del rango esperado [0-9]. "
                  << "Rango actual: [" << min_label << "-" << max_label << "]" << std::endl;
    }
    
    std::cout << "dataset validado correctamente!" << std::endl;
    return true;
}

std::vector<float> MNISTLoader::flattenImage(const std::vector<std::vector<float>>& image) {
    std::vector<float> flattened;
    flattened.reserve(28 * 28);
    
    for (const auto& row : image) {
        for (float pixel : row) {
            flattened.push_back(pixel);
        }
    }
    
    return flattened;
}

void MNISTLoader::normalizeImages(std::vector<std::vector<std::vector<float>>>& images) {
    // Calcular media y desviación estandar global
    float sum = 0.0f;
    int total_pixels = 0;
    
    for (const auto& image : images) {
        for (const auto& row : image) {
            for (float pixel : row) {
                sum += pixel;
                total_pixels++;
            }
        }
    }
    
    float mean = sum / total_pixels;
    
    float variance_sum = 0.0f;
    for (const auto& image : images) {
        for (const auto& row : image) {
            for (float pixel : row) {
                variance_sum += (pixel - mean) * (pixel - mean);
            }
        }
    }
    
    float std_dev = std::sqrt(variance_sum / total_pixels);
    
    // Normalizar todas las imagenes
    for (auto& image : images) {
        for (auto& row : image) {
            for (float& pixel : row) {
                pixel = (pixel - mean) / std_dev;
            }
        }
    }
    
    std::cout << "Imagenes normalizadas - media: " << mean << ", Std: " << std_dev << std::endl;
}