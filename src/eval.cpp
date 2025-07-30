#include "eval.h"
#include "custom_mnist_dataset.h"
#include "vit_module.h"
#include <fstream>

template <typename ModelType, typename DataLoader>
EvalResult evaluate_model(ModelType& model, DataLoader& dataloader, torch::Device device) {
    model->eval();
    torch::NoGradGuard no_grad;

    int correct = 0;
    double total_loss = 0;
    int total = 0;

    for (const auto& batch : dataloader) {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);

        auto output = model->forward(data);
        total_loss += torch::nn::functional::cross_entropy(output, target).template item<double>();

        auto pred = output.argmax(1);
        correct += pred.eq(target).sum().template item<int64_t>();
        total += data.size(0);
    }

    EvalResult result;
    result.accuracy = static_cast<double>(correct) / total;
    result.average_loss = total_loss / total;
    result.total_samples = total;
    return result;
}


template EvalResult evaluate_model<
    ViTModule,
    torch::data::StatelessDataLoader<
        torch::data::datasets::MapDataset<
            CustomMNIST,
            torch::data::transforms::Stack<torch::data::Example<>>
        >,
        torch::data::samplers::SequentialSampler
    >
>(
    ViTModule&,
    torch::data::StatelessDataLoader<
        torch::data::datasets::MapDataset<
            CustomMNIST,
            torch::data::transforms::Stack<torch::data::Example<>>
        >,
        torch::data::samplers::SequentialSampler
    >&,
    torch::Device
);