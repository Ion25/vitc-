#pragma once
#include <torch/torch.h>

struct MultiHeadAttentionImpl : torch::nn::Module {
    MultiHeadAttentionImpl(int embed_dim, int num_heads);
    torch::Tensor forward(torch::Tensor x);

    int num_heads;
    int head_dim;
    torch::nn::Linear qkv_proj{nullptr};
    torch::nn::Linear out_proj{nullptr};
};
TORCH_MODULE(MultiHeadAttention);