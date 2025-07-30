# Vision Transformer (ViT) Implementation in C++

## 👨‍🏫 Integrantes del Proyecto

- **Apaza Condori, Jhon Antony**  
- **Carazas Quispe, Alessander Jesus**  
- **Mariños Hilario, Princce Yorwin**  
- **Mena Quispe, Sergio Sebastian Santos**

---

Este proyecto implementa desde cero un **Vision Transformer (ViT)** en **C++**, sin el uso de frameworks de aprendizaje profundo como PyTorch o TensorFlow. Está diseñado con fines educativos y demuestra cómo un modelo transformer puede adaptarse a tareas de visión por computadora como la clasificación de imágenes, utilizando únicamente operaciones matriciales y lógica de bajo nivel.

El proyecto soporta entrenamiento y predicción sobre los datasets **MNIST** (dígitos manuscritos) y **Fashion-MNIST** (ropa), y permite hacer inferencia con imágenes externas en formato `.jpg`.

---

## 🧠 ¿Qué es un Vision Transformer?

Los **Vision Transformers (ViT)** son una arquitectura que aplica el mecanismo de atención de los transformers (originalmente desarrollados para NLP) directamente sobre imágenes. En lugar de usar convoluciones, dividen la imagen en *patches*, los embeben como vectores y los procesan como una secuencia. Este proyecto implementa esa idea en C++ desde cero.

---

## 📁 Estructura del Proyecto

.
├── main.cpp                     # Entrenamiento en MNIST
├── main_fashion.cpp            # Entrenamiento en Fashion-MNIST
├── predict.cpp                 # Predicción con modelo MNIST
├── predict_fashion.cpp        # Predicción con modelo Fashion-MNIST
├── include/                    # Componentes del modelo ViT
│   ├── vit_module.h
│   ├── patch_embedding.h
│   ├── encoder_block.h
│   ├── multihead_attention.h
│   ├── mlp.h
│   └── utils.h
├── data/                       # Datasets e imágenes de prueba
│   ├── mnist/
│   ├── fashion-mnist/
│   ├── pruebatest.jpg
│   └── pruebaropa.jpg
├── entrenados/                 # Modelos entrenados y estadísticas
│   └── results/, results_fashion/
├── CMakeLists.txt              # Archivo de compilación CMake

---

## ⚙️ Requisitos

- CMake 3.10 o superior
- Compilador con soporte C++17
- Eigen3 (para operaciones matriciales)
- OpenCV (para lectura de imágenes `.jpg`)

### En Ubuntu/Debian:

sudo apt update
sudo apt install cmake libeigen3-dev libopencv-dev g++

---

## 🔧 Compilación

mkdir build
cd build
cmake ..
make

Esto generará ejecutables como `train_mnist`, `train_fashion`, `predict`, y `predict_fashion`.

---

## 🚀 Uso

### Entrenamiento en MNIST

./train_mnist

### Entrenamiento en Fashion-MNIST

./train_fashion

Al finalizar, el modelo entrenado y estadísticas se guardan automáticamente en la carpeta `entrenados/`.

### Predicción con imagen externa

1. tener una imagen en escala de grises de 28x28 px.
2. Ejecuta:

./predict data/pruebatest.jpg
./predict_fashion data/pruebaropa.jpg

Verás una salida con la predicción del modelo sobre la imagen.

---

## 📊 Resultados

Los entrenamientos generan estadísticas en archivos `.csv` (precisión, pérdida por época) dentro de `entrenados/results/`. Esto permite evaluar el rendimiento del modelo en validación y prueba.

---

## 🎓 Motivación

Este proyecto fue creado con fines didácticos para entender los componentes internos de un Vision Transformer, implementando desde cero el flujo de:
- Embedding de patches
- Atención multi-cabeza
- Normalización y capas MLP
- Cálculo de pérdida y entrenamiento

---