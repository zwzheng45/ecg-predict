#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <iostream>

#define STEP 256

using namespace std;

using namespace tensorflow;

int main() {
    // 初始化Session Init session
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        cout << status.ToString() << endl;
        return 1;
    }

    
    // 读取GraphDef Read GraphDef
    GraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), "/Users/zzw/学习/UK/Computer Science/A-level project/心电图识别/model.pb", &graph_def);
    if (!status.ok()) {
        cout << status.ToString() << endl;
        return 1;
    }

    // 将GraphDef添加到Session中 Add GraphDef to Session
    status = session->Create(graph_def);
    if (!status.ok()) {
        cout << status.ToString() << endl;
        return 1;
    }
    
    //从文件读取 Read from binary file
    int cols=1,rows;
    FILE * pfile=fopen("/Users/zzw/学习/UK/Computer Science/A-level project/David A-LEVEL Project/生物传感器/生物信号处理分析算法/projects/基于心电图和机器学习的异常诊断/ecg/ecg-master/ecg/A01621.bin","rb");
    #define ECGSIZE 200000
    short ecgdata[ECGSIZE];
    fseek(pfile,0,SEEK_END);
    int filesize=ftell(pfile);
    fseek(pfile,0,SEEK_SET);
    fread(ecgdata,filesize,1,pfile);
    fclose(pfile);
    rows= filesize/sizeof(short);
    
    //
    rows=STEP*(int)(rows/STEP);
    
    float data[ECGSIZE];
    for(int i=0;i<rows;i++)
    {
        data[i]=ecgdata[i];
    }
    
    // 标准化 Normalization
    // 计算均值和标准差 Calc mean and standard deviation
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float value = data[i * cols + j];
            sum += value;
            sum_sq += value * value;
        }
    }
    double mean = sum / (rows * cols);
    double variance = (sum_sq / (rows * cols)) - (mean * mean);
    double stddev = sqrt(variance);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i * cols + j] = (data[i * cols + j] - mean) / stddev;
        }
    }
    
    
    // 创建输入Tensor Create input tensor
    Tensor input_tensor(DT_FLOAT, TensorShape({1, static_cast<long long>(rows), 1}));
    auto input_tensor_mapped = input_tensor.tensor<float, 3>();

    
    // 填充输入Tensor数据 Fill input tensor with data
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            input_tensor_mapped(0, i, j) = data[i * cols + j];
        }
    }

    // 运行Session Run Session
    std::vector<Tensor> outputs;
    status = session->Run({{"inputs:0", input_tensor}}, {"activation_34/truediv:0"}, {}, &outputs);
    if (!status.ok()) {
        cout << status.ToString() << endl;
        return 1;
    }

    // 输出结果 output
    Tensor output_tensor = outputs[0];
    auto output_tensor_mapped = output_tensor.tensor<float, 3>();

    cout<<outputs[0];
    
    
    // 获取输出张量的形状 Get output tensor shape
    TensorShape output_shape = output_tensor.shape();
    cout << "Output shape: " << output_shape.DebugString() << endl;

    // 获取张量的维度 Get output tensor dimension
    int dim0 = output_tensor_mapped.dimension(0);
    int dim1 = output_tensor_mapped.dimension(1);
    int dim2 = output_tensor_mapped.dimension(2);

    // 输出 final output
    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            for (int k = 0; k < dim2; ++k) {
                cout << "output_tensor_mapped(" << i << ", " << j << ", " << k << ") = "
                     << output_tensor_mapped(i, j, k) << endl;
            }
        }
    }

    session->Close();

    return 0;
}
