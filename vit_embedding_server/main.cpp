#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "crow.h"

// Load the TorchScript model
torch::jit::script::Module load_model(const std::string& model_path) {
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
        std::cout << "Model loaded successfully!" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    return model;
}

// Convert JSON input to a Torch tensor with resizing
torch::Tensor parse_input(const crow::json::rvalue& body) {
    if (!body.has("image")) {
        throw std::runtime_error("Missing 'image' field in JSON.");
    }

    std::vector<float> image_data;
    for (const auto& row : body["image"]) {
        for (const auto& pixel : row) {
            for (const auto& channel : pixel) {
                image_data.push_back(channel.d());
            }
        }
    }

    // Convert vector to a Torch tensor (Assuming original size is unknown, reshape to WxHx3)
    int width = std::sqrt(image_data.size() / 3);
    int height = width;
    torch::Tensor input_tensor = torch::from_blob(image_data.data(), {1, height, width, 3}, torch::kFloat32);

    // Resize to 224x224 using bilinear interpolation
    input_tensor = input_tensor.permute({0, 3, 1, 2}); // Convert to (1,3,H,W)
    input_tensor = torch::nn::functional::interpolate(input_tensor, 
                                                       torch::nn::functional::InterpolateFuncOptions()
                                                           .size(std::vector<int64_t>({224, 224}))
                                                           .mode(torch::kBilinear)
                                                           .align_corners(false));
    return input_tensor.clone();  // Clone to ensure proper ownership
}

int main() {
    crow::SimpleApp app;
    std::string model_path = "/mnt/c/Users/hcang/OneDrive/Desktop/embedding_server_torch/vit_embedding.pt";
    torch::jit::script::Module model = load_model(model_path);

    CROW_ROUTE(app, "/")([]() {
        return crow::response("Send an image array as JSON to /embed.");
    });

    CROW_ROUTE(app, "/embed").methods(crow::HTTPMethod::POST)([&model](const crow::request& req) {
        crow::json::rvalue body;
        try {
            body = crow::json::load(req.body);
        } catch (...) {
            return crow::response(400, "Invalid JSON format.");
        }

        try {
            torch::Tensor input_tensor = parse_input(body);
            torch::Tensor embedding = model.forward({input_tensor}).toTensor();
            std::ostringstream oss;
            oss << embedding;
            return crow::response(oss.str());
        } catch (const std::exception& e) {
            return crow::response(400, e.what());
        }
    });

    app.port(8080).multithreaded().run();
}
