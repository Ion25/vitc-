#pragma once
#include <torch/torch.h>

struct PatchEmbeddingImpl : torch::nn::Module {
    PatchEmbeddingImpl(int img_size, int patch_size, int in_channels, int embed_dim);
    torch::Tensor forward(torch::Tensor x);

    int num_patches;
    torch::nn::Conv2d projection{nullptr};
};
TORCH_MODULE(PatchEmbedding);