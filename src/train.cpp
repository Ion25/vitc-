// train.cpp
#include "train.h"
#include "custom_mnist_dataset.h"
#include "vit_module.h"
#include <torch/torch.h>
#include <fstream>
#include <iomanip>

template <typename DataLoader>
void train_model(ViTModule& model, DataLoader& dataloader, torch::optim::Optimizer& optimizer,
                 int epoch, torch::Device device, const std::string& log_path) {
    model->train();
    float total_loss = 0.0;
    int batch_idx = 0;
    int total_samples = 0;

    std::ofstream file(log_path, std::ios::app);
    if (epoch == 1) {
        file << "epoch,batch,loss\n";
    }

    for (auto& batch : dataloader) {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);

        optimizer.zero_grad();
        auto output = model(data);  // ✅ Llamamos al operador () de ViTModule
        auto loss = torch::nn::functional::cross_entropy(output, target);
        loss.backward();
        optimizer.step();

        float loss_value = loss.template item<float>();
        total_loss += loss_value;
        total_samples += data.size(0);

        std::cout << "[Epoch " << epoch << "][Batch " << batch_idx
                  << "] Loss: " << std::fixed << std::setprecision(4) << loss_value << std::endl;

        file << epoch << "," << batch_idx << "," << loss_value << "\n";
        batch_idx++;
    }

    file.close();
}

// ✅ Instanciación explícita para el tipo específico usado en main.cpp
template void train_model<
    torch::data::StatelessDataLoader<
        torch::data::datasets::MapDataset<
            CustomMNIST,
            torch::data::transforms::Stack<torch::data::Example<>>
        >,
        torch::data::samplers::RandomSampler
    >
>(
    ViTModule&,
    torch::data::StatelessDataLoader<
        torch::data::datasets::MapDataset<
            CustomMNIST,
            torch::data::transforms::Stack<torch::data::Example<>>
        >,
        torch::data::samplers::RandomSampler
    >&,
    torch::optim::Optimizer&,
    int,
    torch::Device,
    const std::string&
);
