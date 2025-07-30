#include "patch_embedding.h"

PatchEmbeddingImpl::PatchEmbeddingImpl(int img_size, int patch_size, int in_channels, int embed_dim) {
    projection = register_module("projection", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, embed_dim, patch_size).stride(patch_size)));
    num_patches = (img_size / patch_size) * (img_size / patch_size);
}

torch::Tensor PatchEmbeddingImpl::forward(torch::Tensor x) {
    x = projection->forward(x);              // [B, D, H', W']
    x = x.flatten(2).transpose(1, 2);        // [B, N, D]
    return x;
}
