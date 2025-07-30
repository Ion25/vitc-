#pragma once
#include <torch/torch.h>
#include "patch_embedding.h"
#include "encoder_block.h"
#include <vector>

class ViTModuleImpl : public torch::nn::Module {
public:
    ViTModuleImpl(int image_size, int patch_size, int in_channels,
                  int embed_dim, int num_heads, int mlp_hidden_dim,
                  int num_layers, int num_classes);

    torch::Tensor forward(torch::Tensor x);

private:
    PatchEmbedding patch_embed;
    std::vector<EncoderBlock> encoder_blocks;
    torch::Tensor cls_token;
    torch::Tensor pos_embed;
    torch::nn::LayerNorm norm{nullptr};
    torch::nn::Linear mlp_head{nullptr};
    int num_patches;
};
TORCH_MODULE(ViTModule);