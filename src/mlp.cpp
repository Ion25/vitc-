#include "mlp.h"

MLPImpl::MLPImpl(int input_dim, int hidden_dim) {
    fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, input_dim));
}

torch::Tensor MLPImpl::forward(torch::Tensor x) {
    return fc2->forward(torch::gelu(fc1->forward(x)));
}
