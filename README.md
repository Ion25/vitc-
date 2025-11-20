
# Vision Transformer (ViT) Implementation in C++

## 👨‍🏫 Integrantes del Proyecto

- Apaza Condori, Jhon Antony  
- Carazas Quispe, Alessander Jesus  
- Mariños Hilario, Princce Yorwin  
- Mena Quispe, Sergio Sebastian Santos  

## 🎯 Objetivo del Proyecto

Implementar desde cero un Vision Transformer (ViT) en C++ usando `libtorch` (PyTorch C++ API), capaz de realizar tareas de clasificación de imágenes en los datasets **MNIST** y **Fashion-MNIST**, incluyendo la posibilidad de predecir imágenes reales y utilizar modelos preentrenados en `.pt`.

## 🏗️ Arquitectura del Proyecto

El proyecto está modularizado en carpetas para código fuente, headers, datos y modelos entrenados.

## 📁 Estructura de Archivos

```
├── include/                    # Archivos de cabecera (Headers)
│   ├── custom_mnist_dataset.h
│   ├── encoder_block.h
│   ├── eval.h
│   ├── mlp.h
│   ├── multihead_attention.h
│   ├── patch_embedding.h
│   ├── stb_image.h
│   ├── stb_image_resize.h
│   ├── train.h
│   ├── utils.h
│   └── vit_module.h
├── src/                        # Implementaciones en C++
│   ├── custom_mnist_dataset.cpp
│   ├── encoder_block.cpp
│   ├── eval.cpp
│   ├── mlp.cpp
│   ├── multihead_attention.cpp
│   ├── patch_embedding.cpp
│   ├── stb_impl.cpp
│   ├── train.cpp
│   ├── utils.cpp
│   └── vit_module.cpp
├── data/                       # Datos para entrenamiento y prueba
│   ├── mnist/                  # MNIST
│   └── fashion-mnist/         # Fashion-MNIST
│   ├── pruebatest.jpg         # Imagen real para MNIST
│   └── pruebaropa.jpg         # Imagen real para Fashion-MNIST
├── entrenados/                # Modelos y resultados entrenados
│   ├── results/
│   └── results_fashion/
├── main.cpp                   # Clasificación con MNIST
├── main_fashion.cpp          # Clasificación con Fashion-MNIST
├── predict.cpp               # Predicción de imagen real (MNIST)
├── predict_fashion.cpp       # Predicción de imagen real (Fashion-MNIST)
├── CMakeLists.txt            # Configuración de compilación
└── README.md
```

## 🧠 Arquitectura del ViT Implementada

- División de imagen en **patches**.
- Embedding de cada patch mediante capa lineal.
- Uso de token `[class]` al inicio.
- Agregado de codificación posicional.
- Encoder con bloques Transformer (Multi-Head Attention + MLP + LayerNorm).
- Capa final de clasificación.

## 🛠️ Compilación

Requiere tener instalado libtorch y CMake.

```bash
mkdir build
cd build
cmake ..
make
```

## 🚀 Ejecución

### Entrenar (ya entrenado en Python, pero se puede cargar modelo `.pt`)

```bash
./main
./main_fashion
```

### Predicción con imagen externa

```bash
./predict data/pruebatest.jpg
./predict_fashion data/pruebaropa.jpg
```

## 📊 Resultados de Entrenamiento

- Accuracy MNIST: ~88%
- Accuracy Fashion-MNIST: ~83%
- Entrenamiento realizado previamente con `train.py`, modelos guardados en:

![WhatsApp Image 2025-07-30 at 1 43 05 PM](https://github.com/user-attachments/assets/6728e017-7465-4dae-bf0f-0a6f1a781f76)
![WhatsApp Image 2025-07-30 at 1 45 56 PM](https://github.com/user-attachments/assets/b99985a8-e1fb-4537-807c-b44711fb8b56)
![WhatsApp Image 2025-07-30 at 1 45 57 PM](https://github.com/user-attachments/assets/bc66933e-6761-41dd-aec4-653c6c96794f)



```
entrenados/results/           # MNIS![Uploading WhatsApp Image 2025-07-30 at 1.43.05 PM.jpeg…]()
T
entrenados/results_fashion/  # Fashion-MNIST
```


Se incluyen estadísticas (`train_stats.csv`, `eval_stats.csv`) y los modelos (`.pt`) entrenados con PyTorch.

## 🔮 Posibilidades Futuras

- Entrenamiento directamente desde C++ (usando `train.h` y `train.cpp`)
- Extensión a CIFAR-10 o ImageNet
- Visualización de mapas de atención
- Uso en otras plataformas o dispositivos embebidos

## 📚 Referencias

- Paper: *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*
- Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/) y [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

## RAMA 1
