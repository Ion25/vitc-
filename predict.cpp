// predict.cpp
#include <torch/torch.h>
#include "vit_module.h"
#include "stb_image.h"
#include "stb_image_resize.h"
#include <iostream>
#include <vector>

int main() {
    std::string model_path = "../entrenados/results/20250730_102921/vit_trained.pt";
    std::string image_path = "../data/pruebatest.jpg";

    // Crear modelo igual al entrenado
    ViTModule model(28, 7, 1, 64, 4, 256, 6, 10); 

    try {
        torch::load(model, model_path);  // ← Aquí carga el state_dict
    } catch (const c10::Error& e) {
        std::cerr << "❌ Error al cargar el modelo: " << e.what() << std::endl;
        return -1;
    }

    model->eval();
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    model->to(device);

    // Cargar y procesar imagen
    int width, height, channels;
    unsigned char* img_data = stbi_load(image_path.c_str(), &width, &height, &channels, 1);
    if (!img_data) {
        std::cerr << "❌ No se pudo cargar la imagen." << std::endl;
        return -1;
    }

    const int target_w = 28, target_h = 28;
    std::vector<unsigned char> resized_img(target_w * target_h);

    if (width != 28 || height != 28) {
        if (!stbir_resize_uint8(img_data, width, height, 0,
                                resized_img.data(), target_w, target_h, 0, 1)) {
            std::cerr << "❌ Error al redimensionar la imagen." << std::endl;
            stbi_image_free(img_data);
            return -1;
        }
        stbi_image_free(img_data);
        img_data = resized_img.data();
    }

    torch::Tensor img_tensor = torch::from_blob(img_data, {1, 1, 28, 28}, torch::kUInt8).to(torch::kFloat32) / 255.0;
    img_tensor = img_tensor.to(device);

    torch::Tensor output = model->forward(img_tensor);
    int pred_label = output.argmax(1).item<int>();

    std::cout << "🔍 Predicción: " << pred_label << std::endl;
    return 0;
}
