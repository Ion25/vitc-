#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <algorithm>
#include "transformer.h"
#include "trainer.h"
#include "matrix.h"

int main() {
    try {
        std::cout << "=== TRANSFORMER ===" << std::endl;
        std::cout << "=============================================" << std::endl;
        
        // PARAMETROS
        const int IMG_SIZE = 28;
        const int PATCH_SIZE = 7;        
        const int D_MODEL = 256;         
        const int NUM_HEADS = 8;         
        const int NUM_LAYERS = 4;        
        const int D_FF = 512;        
        const int NUM_CLASSES = 10;
        const float LEARNING_RATE = 0.0003f;
        const int NUM_EPOCHS = 20;       // ÉPOCAS
        
        // Crear modelo
        VisionTransformer model(IMG_SIZE, PATCH_SIZE, D_MODEL, NUM_HEADS, 
                       NUM_LAYERS, D_FF, NUM_CLASSES, true);
        
        std::cout << "Modelo creado exitosamente!" << std::endl;
        std::cout << "\nConfiguracion OPTIMIZADA:" << std::endl;
        std::cout << "   - Imagen: " << IMG_SIZE << "x" << IMG_SIZE << std::endl;
        std::cout << "   - Patch: " << PATCH_SIZE << "x" << PATCH_SIZE << std::endl;
        std::cout << "   - Patches: " << (IMG_SIZE/PATCH_SIZE) * (IMG_SIZE/PATCH_SIZE) << std::endl;
        std::cout << "   - Dimension: " << D_MODEL << " (aumentada)" << std::endl;
        std::cout << "   - Cabezas: " << NUM_HEADS << " (mas atencion)" << std::endl;
        std::cout << "   - Capas: " << NUM_LAYERS << " (mas profundidad)" << std::endl;
        std::cout << "   - Learning Rate: " << LEARNING_RATE << " (muy bajo)" << std::endl;
        std::cout << "   - Epocas: " << NUM_EPOCHS << " (mas entrenamiento)" << std::endl;
        //  CARGAR MODELO ENTRENADO
        std::cout << "\n=== Buscando Modelo Entrenado ===" << std::endl;
        bool model_loaded = model.loadModel("vit_mnist_final");
        
        if (model_loaded) {
            std::cout << "Modelo entrenado encontrado y cargado!" << std::endl;
        } else {
            std::cout << "No se encontro modelo entrenado. Se entrenara uno nuevo." << std::endl;
        }
        
        // CARGAR DATOS
        std::cout << "\n=== Cargando Datos ===" << std::endl;
        
        std::vector<std::string> possible_files = {
            "fashion-mnist_test.csv",
            "fashion-mnist_train.csv",
            "mnist_train.csv",
            "mnist_test.csv"
        };
        
        std::vector<std::vector<std::vector<float>>> train_images, test_images;
        std::vector<int> train_labels, test_labels;
        
        for (const auto& filename : possible_files) {
            std::cout << "Buscando: " << filename << std::endl;
            if (filename.find("train") != std::string::npos) {
                auto [imgs, lbls] = MNISTLoader::loadMNISTCSV(filename, 60000);
                if (!imgs.empty()) {
                    train_images = imgs;
                    train_labels = lbls;
                    std::cout << "Datos de entrenamiento: " << filename << std::endl;
                    break;
                }
            }
        }
        
        for (const auto& filename : possible_files) {
            if (filename.find("test") != std::string::npos) {
                auto [imgs, lbls] = MNISTLoader::loadMNISTCSV(filename, 10000);
                if (!imgs.empty()) {
                    test_images = imgs;
                    test_labels = lbls;
                    std::cout << "Datos de prueba: " << filename << std::endl;
                    break;
                }
            }
        }
        
        if (train_images.empty() && !test_images.empty()) {
            std::cout << "Usando datos de test tambien para entrenamiento" << std::endl;
            train_images = test_images;
            train_labels = test_labels;
        } else if (!train_images.empty() && test_images.empty()) {
            std::cout << "Usando datos de entrenamiento tambien para prueba" << std::endl;
            test_images = train_images;
            test_labels = train_labels;
        }
        
        if (train_images.empty() || test_images.empty()) {
            std::cout << "No se pudieron cargar datos CSV. Programa terminado." << std::endl;
            return 1;
        }
        
        std::cout << "\nDatos cargados:" << std::endl;
        std::cout << "   - Entrenamiento: " << train_images.size() << " imagenes" << std::endl;
        std::cout << "   - Prueba: " << test_images.size() << " imagenes" << std::endl;
        
        std::cout << "\nAplicando data augmentation controlado..." << std::endl;
        std::vector<std::vector<std::vector<float>>> augmented_images = train_images;
        std::vector<int> augmented_labels = train_labels;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> noise(0.0f, 0.02f); 
        
        for (size_t i = 0; i < train_images.size(); i++) {
            std::vector<std::vector<float>> noisy_image = train_images[i];
            for (int r = 0; r < 28; r++) {
                for (int c = 0; c < 28; c++) {
                    noisy_image[r][c] += noise(gen);
                    noisy_image[r][c] = std::max(0.0f, std::min(1.0f, noisy_image[r][c]));
                }
            }
            augmented_images.push_back(noisy_image);
            augmented_labels.push_back(train_labels[i]);
        }
        
        train_images = augmented_images;
        train_labels = augmented_labels;
        std::cout << "Dataset aumentado a " << train_images.size() << " imagenes" << std::endl;
        
        //  EVALUACION INICIAL
        std::cout << "\n=== Evaluacion Inicial ===" << std::endl;
        float initial_accuracy = evaluateModel(model, test_images, test_labels);
        std::cout << "Precision inicial: " << std::fixed << std::setprecision(1) 
                  << initial_accuracy << "%" << std::endl;
        
        //  ENTRENAMIENTO 
        if (!model_loaded || initial_accuracy < 85.0f) {
            
            auto start_time = std::chrono::high_resolution_clock::now();
            float best_accuracy = initial_accuracy;
            int patience_counter = 0;
            const int PATIENCE = 8;
            
            for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
                std::cout << "\nEpoca " << (epoch + 1) << "/" << NUM_EPOCHS << std::endl;
                std::cout << std::string(50, '-') << std::endl;
                
                float current_lr = LEARNING_RATE;
                
                if (epoch >= 30) current_lr *= 0.8f;
                if (epoch >= 45) current_lr *= 0.6f;
                
                std::cout << "Learning rate: " << std::fixed << std::setprecision(6) << current_lr << std::endl;
                
                // Mezclar datos
                std::vector<int> indices(train_images.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), gen);
                
                // Entrenar con progreso detallado
                for (size_t idx = 0; idx < train_images.size(); idx++) {
                    int i = indices[idx];
                    model.train(train_images[i], train_labels[i], current_lr);
                    
                    // Mostrar progreso 
                    if ((idx + 1) % 200 == 0 || idx == train_images.size() - 1) {
                        float progress = (float)(idx + 1) / train_images.size() * 100;
                        
                        // Evaluacion intermedia
                        if ((idx + 1) % 400 == 0) {
                            float interim_acc = evaluateModel(model, 
                                std::vector<std::vector<std::vector<float>>>(test_images.begin(), test_images.begin() + 100),
                                std::vector<int>(test_labels.begin(), test_labels.begin() + 100));
                            std::cout << "  Progreso: " << std::setprecision(1) << progress 
                                      << "% - Precision parcial: " << interim_acc << "%" << std::endl;
                        } else {
                            std::cout << "  Progreso: " << std::setprecision(1) << progress << "%" << std::endl;
                        }
                    }
                }
                
                // Evaluacion completa de la epoca
                float epoch_accuracy = evaluateModel(model, test_images, test_labels);
                std::cout << "Precision al final de epoca " << (epoch + 1) << ": " 
                          << std::setprecision(1) << epoch_accuracy << "%" << std::endl;
                
               
                if (epoch_accuracy > best_accuracy + 0.2f) { 
                    best_accuracy = epoch_accuracy;
                    patience_counter = 0;
                    model.saveModel("vit_mnist_final");
                    std::cout << "*** NUEVO MEJOR MODELO! (" << best_accuracy << "%) ***" << std::endl;
                } else {
                    patience_counter++;
                    std::cout << "Sin mejora significativa (" << patience_counter << "/" << PATIENCE << ")" << std::endl;
                    
                    if (patience_counter >= PATIENCE) {
                        std::cout << "Early stopping: sin mejora en " << PATIENCE << " epocas" << std::endl;
                        break;
                    }
                }
                
                if (epoch_accuracy >= 90.0f) {
                    std::cout << "*** OBJETIVO ALCANZADO: 90%+ DE PRECISION! ***" << std::endl;
                    break;
                }
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
            
            std::cout << "\nTiempo total de entrenamiento: " << duration.count() << " segundos" << std::endl;
            std::cout << "Mejor precision alcanzada: " << best_accuracy << "%" << std::endl;
            
        } else {
            std::cout << "\nModelo ya entrenado con excelente precision. Saltando entrenamiento." << std::endl;
        }
        
        // 6. EVALUACION FINAL DETALLADA
        std::cout << "\n=== Evaluacion Final Completa ===" << std::endl;
        
        int correct = 0;
        int total = (int)test_images.size();
        std::vector<int> class_correct(10, 0);
        std::vector<int> class_total(10, 0);
        
        std::cout << "Evaluando " << total << " imagenes de muestra detallada:" << std::endl;
        std::cout << std::endl;
        
        for (int i = 0; i < total; i++) {
            Matrix output = model.forward(test_images[i], false);
            Matrix probs = output.softmax();
            
            int predicted = 0;
            float max_prob = probs.data[0][0];
            for (int j = 1; j < probs.cols; j++) {
                if (probs.data[0][j] > max_prob) {
                    max_prob = probs.data[0][j];
                    predicted = j;
                }
            }
            
            bool is_correct = (predicted == test_labels[i]);
            if (is_correct) {
                correct++;
                class_correct[test_labels[i]]++;
            }
            class_total[test_labels[i]]++;
            
            std::cout << "Imagen " << std::setw(2) << (i+1) << ": "
                      << "Real=" << test_labels[i] << ", "
                      << "Predicho=" << predicted << ", "
                      << "Confianza=" << std::fixed << std::setprecision(1) << (max_prob * 100) << "% "
                      << (is_correct ? "✓" : "✗") << std::endl;
        }
        
        float final_accuracy = evaluateModel(model, test_images, test_labels);
        
        std::cout << "\n === RESULTADOS FINALES ===" << std::endl;
        std::cout << "Precision en muestra detallada: " << correct << "/" << total 
                  << " (" << std::fixed << std::setprecision(1) << ((float)correct/total*100) << "%)" << std::endl;
        std::cout << " PRECISION TOTAL FINAL: " << std::setprecision(1) << final_accuracy << "%" << std::endl;
        
        // Analisis por clase
        std::cout << "\n Precision por clase:" << std::endl;
        for (int i = 0; i < 10; i++) {
            if (class_total[i] > 0) {
                float class_acc = (float)class_correct[i] / class_total[i] * 100;
                std::cout << "   Clase " << i << ": " << std::setprecision(1) 
                          << class_acc << "% (" << class_correct[i] << "/" << class_total[i] << ")" << std::endl;
            }
        }
        
        // Evaluacion del resultado
        if (final_accuracy >= 90.0f) {
            std::cout << "\nOBJETIVO CUMPLIDO: 90%+ DE PRECISION!" << std::endl;
            std::cout << "El Vision Transformer esta funcionando perfectamente." << std::endl;
        } else if (final_accuracy >= 80.0f) {
            std::cout << "\n  Muy cerca del objetivo (90%)." << std::endl;
            std::cout << "Ejecuta de nuevo para continuar entrenando." << std::endl;
        } else if (final_accuracy >= 70.0f) {
            std::cout << "\nEl modelo esta aprendiendo bien." << std::endl;
            std::cout << "Necesita mas entrenamiento para llegar al 90%." << std::endl;
        } else {
            std::cout << "\n El modelo necesita mas optimizacion." << std::endl;
        }
                
        std::cout << "\n === Archivos del Modelo Final ===" << std::endl;
        std::cout << "Modelo guardado como:" << std::endl;
        std::cout << "  - vit_mnist_final_*.bin (todos los pesos)" << std::endl;
        
        std::cout << "\n Vision Transformer completado exitosamente!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}