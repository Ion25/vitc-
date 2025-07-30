#pragma once
#include <torch/torch.h>

struct MLPImpl : torch::nn::Module {
    MLPImpl(int input_dim, int hidden_dim);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};
TORCH_MODULE(MLP);