cmake_minimum_required(VERSION 3.10)

project(ViTEmbeddingServer)

# ✅ Force C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ✅ Set the correct compiler
set(CMAKE_C_COMPILER /usr/bin/gcc-11)
set(CMAKE_CXX_COMPILER /usr/bin/g++-11)

# ✅ Set LibTorch path
set(CMAKE_PREFIX_PATH "{PATH_TO_LIBTORCH}/libtorch")

# ✅ Find necessary packages
find_package(Torch REQUIRED)
find_package(Crow CONFIG REQUIRED)
find_package(OpenSSL REQUIRED) 

add_executable(vit_embedding_server main.cpp)

# ✅ Link libraries
target_link_libraries(vit_embedding_server "${TORCH_LIBRARIES}" Crow::Crow
    OpenSSL::SSL OpenSSL::Crypto
    -lz)