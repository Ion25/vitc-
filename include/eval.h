#pragma once
#include <torch/torch.h>
#include <string>

struct EvalResult {
    double accuracy;
    double average_loss;
    int total_samples;
};

//template <typename DataLoader>
//EvalResult evaluate_model(torch::nn::Module& model, DataLoader& dataloader, torch::Device device);

template <typename ModelType, typename DataLoader>
EvalResult evaluate_model(ModelType& model, DataLoader& dataloader, torch::Device device);
