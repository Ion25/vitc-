#pragma once
#include <string>
#include <torch/torch.h>
#include "eval.h"
#include "vit_module.h"  // Necesario para saber qué es ViTModule

void save_model(const ViTModule& model, const std::string& path);
void load_model(ViTModule& model, const std::string& path);

std::string timestamp();
void save_eval_results_csv(const std::string& filename, int epoch, const EvalResult& result);
