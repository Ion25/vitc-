#include "custom_mnist_dataset.h"
#include <fstream>
#include <cassert>
#include <vector>
#include <zlib.h>

static uint32_t read_u32(std::ifstream& f) {
    uint32_t v;
    f.read(reinterpret_cast<char*>(&v), 4);
    return __builtin_bswap32(v);
}

static torch::Tensor read_images(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    assert(f && "Cannot open image file");
    uint32_t magic = read_u32(f);
    uint32_t count = read_u32(f);
    uint32_t rows = read_u32(f);
    uint32_t cols = read_u32(f);
    std::vector<uint8_t> buffer(rows * cols * count);
    f.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    torch::Tensor t = torch::from_blob(buffer.data(), {static_cast<int64_t>(count), 1, (int)rows, (int)cols}, torch::kUInt8).to(torch::kFloat32).div(255);
    return t.clone();
}

static torch::Tensor read_labels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    assert(f && "Cannot open label file");
    uint32_t magic = read_u32(f);
    uint32_t count = read_u32(f);
    std::vector<uint8_t> buffer(count);
    f.read(reinterpret_cast<char*>(buffer.data()), count);
    return torch::from_blob(buffer.data(), {static_cast<int64_t>(count)}, torch::kUInt8).to(torch::kLong).clone();
}

CustomMNIST::CustomMNIST(const std::string& images_path,
                         const std::string& labels_path,
                         bool /*train*/) {
    images_ = read_images(images_path);
    labels_ = read_labels(labels_path);
}

torch::data::Example<> CustomMNIST::get(size_t idx) {
    return {images_[idx], labels_[idx]};
}

torch::optional<size_t> CustomMNIST::size() const {
    return images_.size(0);
}
