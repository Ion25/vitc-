#include "multihead_attention.h"

MultiHeadAttentionImpl::MultiHeadAttentionImpl(int embed_dim, int num_heads)
    : num_heads(num_heads), head_dim(embed_dim / num_heads) {
    qkv_proj = register_module("qkv_proj", torch::nn::Linear(embed_dim, embed_dim * 3));
    out_proj = register_module("out_proj", torch::nn::Linear(embed_dim, embed_dim));
}

torch::Tensor MultiHeadAttentionImpl::forward(torch::Tensor x) {
    auto B = x.size(0), N = x.size(1), D = x.size(2);
    auto qkv = qkv_proj->forward(x).chunk(3, -1); // 3x [B, N, D]
    auto q = qkv[0].view({B, N, num_heads, head_dim}).transpose(1, 2); // [B, k, N, d_k]
    auto k = qkv[1].view({B, N, num_heads, head_dim}).transpose(1, 2);
    auto v = qkv[2].view({B, N, num_heads, head_dim}).transpose(1, 2);

    auto attn = torch::softmax(torch::matmul(q, k.transpose(-2, -1)) / std::sqrt((double)head_dim), -1);
    auto out = torch::matmul(attn, v).transpose(1, 2).reshape({B, N, D});
    return out_proj->forward(out);
}
