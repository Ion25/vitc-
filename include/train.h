#pragma once
#include <torch/torch.h>
#include <string>
#include "vit_module.h"

template <typename DataLoader>
void train_model(ViTModule& model, DataLoader& dataloader,
                 torch::optim::Optimizer& optimizer, int epoch,
                 torch::Device device, const std::string& log_path);
