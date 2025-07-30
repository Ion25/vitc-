#include "vit_module.h"

ViTModuleImpl::ViTModuleImpl(int image_size, int patch_size, int in_channels,
                             int embed_dim, int num_heads, int mlp_hidden_dim,
                             int num_layers, int num_classes)
    : patch_embed(image_size, patch_size, in_channels, embed_dim) {

    num_patches = patch_embed->num_patches;
    cls_token = register_parameter("cls_token", torch::zeros({1, 1, embed_dim}));
    pos_embed = register_parameter("pos_embed", torch::zeros({1, num_patches + 1, embed_dim}));
    //norm = register_module("norm", torch::nn::LayerNorm(embed_dim));
    norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
    mlp_head = register_module("mlp_head", torch::nn::Linear(embed_dim, num_classes));
    register_module("patch_embed", patch_embed);

    for (int i = 0; i < num_layers; ++i) {
        encoder_blocks.push_back(EncoderBlock(embed_dim, num_heads, mlp_hidden_dim));
        register_module("encoder_block_" + std::to_string(i), encoder_blocks.back());
    }
}

torch::Tensor ViTModuleImpl::forward(torch::Tensor x) {
    x = patch_embed->forward(x);                         // [B, N, D]
    auto B = x.size(0);
    auto cls = cls_token.expand({B, -1, -1});            // [B, 1, D]
    x = torch::cat({cls, x}, 1);                         // [B, N+1, D]
    x = x + pos_embed;
    for (auto& block : encoder_blocks)
        x = block->forward(x);
    x = norm->forward(x);
    return mlp_head->forward(x.select(1, 0));            // [B, C]
}
