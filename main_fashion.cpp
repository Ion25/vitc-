// main_fashion.cpp
#include "vit_module.h"
#include "custom_mnist_dataset.h"
#include "train.h"
#include "eval.h"
#include "utils.h"
#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <chrono>

int main() {
    torch::manual_seed(42);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA disponible. GPUs detectadas: " << torch::cuda::device_count() << std::endl;
        std::cout << "Usando dispositivo: " << device << std::endl;
    } else {
        std::cout << "CUDA no disponible. Usando CPU." << std::endl;
    }


    // ────────────────────────────────
    // Configuración de dataset (Fashion-MNIST)
    // ────────────────────────────────
    std::string data_dir = "../data/fashion-mnist";
    auto train_dataset = CustomMNIST(
        data_dir + "/train-images-idx3-ubyte",
        data_dir + "/train-labels-idx1-ubyte")
        .map(torch::data::transforms::Stack<>());

    auto test_dataset = CustomMNIST(
        data_dir + "/t10k-images-idx3-ubyte",
        data_dir + "/t10k-labels-idx1-ubyte")
        .map(torch::data::transforms::Stack<>());

    const int batch_size = 64;

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    // ────────────────────────────────
    // Inicialización del modelo ViT
    // ────────────────────────────────
    ViTModule model(/*image_size=*/28, /*patch_size=*/7, /*in_channels=*/1,
                    /*embed_dim=*/64, /*num_heads=*/4, /*mlp_hidden_dim=*/256,
                    /*num_layers=*/6, /*num_classes=*/10);
    model->to(device);

    torch::optim::Adam optimizer(model->parameters(),
        torch::optim::AdamOptions(1e-3).weight_decay(0.1));

    // ────────────────────────────────
    // Ciclo de entrenamiento + evaluación
    // ────────────────────────────────
    auto start_time = std::chrono::high_resolution_clock::now();
    const int num_epochs = 20;
    std::string run_id = timestamp();
    std::filesystem::create_directories("results_fashion/" + run_id);

    std::string train_log = "results_fashion/" + run_id + "/train_stats.csv";
    std::string eval_log = "results_fashion/" + run_id + "/eval_stats.csv";
    std::string model_path = "results_fashion/" + run_id + "/vit_fashion.pt";

    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        train_model(model, *train_loader, optimizer, epoch, device, train_log);

        auto result = evaluate_model(model, *test_loader, device);
        std::cout << "[Epoch " << epoch << "] Accuracy: " << result.accuracy * 100 << "%"
                  << ", Avg Loss: " << result.average_loss << std::endl;

        save_eval_results_csv(eval_log, epoch, result);
    }

    // ────────────────────────────────
    // Guardar modelo final
    // ────────────────────────────────
    save_model(model, model_path);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    auto total_seconds = static_cast<int>(elapsed_seconds.count());
    int minutes = total_seconds / 60;
    int seconds = total_seconds % 60;

    std::cout << "Entrenamiento completado en "
            << minutes << " minutos y "
            << seconds << " segundos." << std::endl;

    return 0;
}
