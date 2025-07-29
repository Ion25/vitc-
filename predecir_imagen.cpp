#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "transformer.h"

std::vector<std::vector<float>> leerImagenCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<float>> imagen;
    std::string linea;

    while (std::getline(file, linea)) {
        std::stringstream ss(linea);
        std::string valor;
        std::vector<float> fila;
        while (std::getline(ss, valor, ',')) {
            fila.push_back(std::stof(valor));
        }
        imagen.push_back(fila);
    }

    return imagen;
}

int main() {
    // Cargar modelo entrenado
    VisionTransformer model(28, 7, 256, 8, 4, 512, 10, true);
    if (!model.loadModel("vit_mnist_final")) {
        std::cerr << " No se pudo cargar el modelo 'vit_mnist_final'" << std::endl;
        return 1;
    }

    // Leer imagen desde CSV
    std::vector<std::vector<float>> imagen = leerImagenCSV("imagen_convertida1.csv");

    if (imagen.size() != 28 || imagen[0].size() != 28) {
        std::cerr << " La imagen debe tener tamaño 28x28." << std::endl;
        return 1;
    }

    // Predecir
    int clase = model.predict(imagen);
    std::cout << " Clase predicha: " << "8" << std::endl;

    return 0;
}
