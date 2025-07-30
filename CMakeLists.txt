cmake_minimum_required(VERSION 3.10)
project(ViTLibTorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/gpu")

# Habilita mensajes coloridos del compilador y optimización básica
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Encuentra la instalación de LibTorch
find_package(Torch REQUIRED)

# Incluir encabezados personalizados
include_directories(${PROJECT_SOURCE_DIR}/include)

# ─────────────────────────────────────────────
# 🔧 Ejecutable principal: main.cpp
# ─────────────────────────────────────────────
file(GLOB_RECURSE SOURCES_MAIN
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/main.cpp"
)

add_executable(train_mnist ${SOURCES_MAIN})
target_link_libraries(train_mnist "${TORCH_LIBRARIES}")
set_property(TARGET train_mnist PROPERTY CXX_STANDARD 17)

# ─────────────────────────────────────────────
# 👕 Ejecutable para Fashion-MNIST: main_fashion.cpp
# ─────────────────────────────────────────────
file(GLOB_RECURSE SOURCES_FASHION
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/main_fashion.cpp"
)

add_executable(train_fashion ${SOURCES_FASHION})
target_link_libraries(train_fashion "${TORCH_LIBRARIES}")
set_property(TARGET train_fashion PROPERTY CXX_STANDARD 17)

# ─────────────────────────────────────────────
# 🔮 Ejecutable de predicción: predict.cpp
# ─────────────────────────────────────────────
file(GLOB_RECURSE SOURCES_PREDICT
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/predict.cpp"
)

add_executable(predict ${SOURCES_PREDICT})
target_link_libraries(predict "${TORCH_LIBRARIES}")
set_property(TARGET predict PROPERTY CXX_STANDARD 17)


# ─────────────────────────────────────────────
# 👗 Ejecutable de predicción para Fashion: predict_fashion.cpp
# ─────────────────────────────────────────────
file(GLOB_RECURSE SOURCES_PREDICT_FASHION
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/predict_fashion.cpp"
)

add_executable(predict_fashion ${SOURCES_PREDICT_FASHION})
target_link_libraries(predict_fashion "${TORCH_LIBRARIES}")
set_property(TARGET predict_fashion PROPERTY CXX_STANDARD 17)
