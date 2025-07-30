#include "encoder_block.h"

EncoderBlockImpl::EncoderBlockImpl(int embed_dim, int num_heads, int mlp_hidden_dim)
    : attention(embed_dim, num_heads), mlp(embed_dim, mlp_hidden_dim) {
    //norm1 = register_module("norm1", torch::nn::LayerNorm(embed_dim));
    //norm2 = register_module("norm2", torch::nn::LayerNorm(embed_dim));
    norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
    norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));

    register_module("attention", attention);
    register_module("mlp", mlp);
}

torch::Tensor EncoderBlockImpl::forward(torch::Tensor x) {
    auto h = x;
    x = norm1->forward(x);
    x = attention->forward(x);
    x = x + h;

    h = x;
    x = norm2->forward(x);
    x = mlp->forward(x);
    x = x + h;
    return x;
}
