#pragma once
#include <torch/torch.h>
#include "multihead_attention.h"
#include "mlp.h"

struct EncoderBlockImpl : torch::nn::Module {
    EncoderBlockImpl(int embed_dim, int num_heads, int mlp_hidden_dim);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    MultiHeadAttention attention;
    MLP mlp;
};
TORCH_MODULE(EncoderBlock);