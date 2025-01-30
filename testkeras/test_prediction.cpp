#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <iostream>

using namespace tensorflow;

int main4() {
    // Init session
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return 1;
    }

    // Load model
    GraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), "/Users/zzw/学习/UK/Computer Science/A-level project/心电图识别/model.pb", &graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return 1;
    }

    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return 1;
    }
    std::cout<<"Session crerated"<<std::endl;

    // Create empty input tensor with 0
    Tensor input_tensor(DT_FLOAT, TensorShape({1, 256, 1}));
    auto input_tensor_mapped = input_tensor.tensor<float, 3>();

    for (int i = 0; i < 256; ++i) {
        for (int j = 0; j < 1; ++j) {
            input_tensor_mapped(0, i, j) = 0.0;
        }
    }


    std::vector<Tensor> outputs;
    status = session->Run({{"inputs:0", input_tensor}}, {"activation_34/truediv:0"}, {}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return 1;
    }

    Tensor output_tensor = outputs[0];
    auto output_tensor_mapped = output_tensor.tensor<float, 3>();

    // 输出每个值及其索引
    TensorShape output_shape = output_tensor.shape();
    int dim0 = output_tensor_mapped.dimension(0);
    int dim1 = output_tensor_mapped.dimension(1);
    int dim2 = output_tensor_mapped.dimension(2);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            for (int k = 0; k < dim2; ++k) {
                std::cout << "output_tensor_mapped(" << i << ", " << j << ", " << k << ") = "
                          << output_tensor_mapped(i, j, k) << std::endl;
            }
        }
    }

    // 关闭Session
    session->Close();

    return 0;
}
