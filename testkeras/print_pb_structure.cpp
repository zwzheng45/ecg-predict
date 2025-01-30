#include <iostream>
#include <fstream>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/platform/env.h>

using namespace tensorflow;

void PrintGraphStructure(const GraphDef& graph_def) {
    for (const auto& node : graph_def.node()) {
        std::cout << "Node name: " << node.name() << std::endl;
        std::cout << "  Op: " << node.op() << std::endl;
        std::cout << "  Inputs: ";
        for (const auto& input : node.input()) {
            std::cout << input << " ";
        }
        std::cout << std::endl;
    }
}

int main3(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "incorrect parameters" << std::endl;
        return -1;
    }

    const std::string model_path = argv[1];

    // Create a new TensorFlow session
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        std::cerr << "error creating session: " << status.ToString() << std::endl;
        return -1;
    }

    // Read model
    GraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
    if (!status.ok()) {
        std::cerr << "error reading graph definition from " << model_path << ": " << status.ToString() << std::endl;
        return -1;
    }

    PrintGraphStructure(graph_def);

    session->Close();

    return 0;
}
