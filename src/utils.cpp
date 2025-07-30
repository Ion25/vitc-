#include "utils.h"
#include <torch/torch.h>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <iostream>
#include <fstream>

void save_model(const ViTModule& model, const std::string& path) {
    torch::save(model, path);
    std::cout << "✅ Modelo guardado en " << path << std::endl;
}

void load_model(ViTModule& model, const std::string& path) {
    torch::load(model, path);
    std::cout << "📂 Modelo cargado desde " << path << std::endl;
}


std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm* tm_ptr = std::localtime(&time);
    std::ostringstream oss;
    oss << std::put_time(tm_ptr, "%Y%m%d_%H%M%S");
    return oss.str();
}

void save_eval_results_csv(const std::string& filename, int epoch, const EvalResult& result) {
    std::ofstream file(filename, std::ios::app);  // modo append
    if (file.is_open()) {
        if (file.tellp() == 0) { // si el archivo está vacío, escribe encabezado
            file << "Epoch,Accuracy,AverageLoss,TotalSamples\n";
        }
        file << epoch << ","
             << result.accuracy << ","
             << result.average_loss << ","
             << result.total_samples << "\n";
        file.close();
    } else {
        std::cerr << "❌ No se pudo abrir " << filename << " para escribir resultados de evaluación.\n";
    }
}
