#pragma once
#include <torch/torch.h>
#include <string>

class CustomMNIST : public torch::data::Dataset<CustomMNIST> {
public:
    CustomMNIST(const std::string& images_path,
                const std::string& labels_path,
                bool train = true);

    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;

private:
    torch::Tensor images_, labels_;
};
