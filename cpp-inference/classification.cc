#include <cstdio>
#include <bits/stdc++.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/core/ocl.hpp>


using namespace cv;
using namespace std;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "classification <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];
  int model_height,model_width,model_channels;
  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Intrepter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  //For gpu tensors
  auto* delegate = TfLiteGpuDelegateV2Create(nullptr);
  if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

  // Allocate tensor buffers.
  //TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  //printf("=== Pre-invoke Interpreter State ===\n");
  //tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

  // Run inference
  //TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  //printf("\n\n=== Post-invoke Interpreter State ===\n");
  //tflite::PrintInterpreterState(interpreter.get());
  interpreter->SetAllowFp16PrecisionForFp32(true);
    //interpreter->SetNumThreads(4);      //quad core

    // Get input dimension from the input tensor metadata
    // Assuming one input only
    int In;
    In = interpreter->inputs()[0];
    model_height   = interpreter->tensor(In)->dims->data[1];
    model_width    = interpreter->tensor(In)->dims->data[2];
    model_channels = interpreter->tensor(In)->dims->data[3];

    cout << "height   : "<< model_height << endl;
    cout << "width    : "<< model_width << endl;
    cout << "channels : "<< model_channels << endl;

    chrono::steady_clock::time_point Tbegin, Tend;
    cv::Mat img;
    
    img=imread("/home/asad/projs/cpp-inference/test.jpg");  //need to refresh frame before dnn class detection
    
    if (img.empty()) {
        cerr << "Can not load picture!" << endl;
        exit(-1);
    }
    
    // copy image to input as input tensor
    // Preprocess the input image to feed to the model
    cv::resize(img, img, Size(model_width,model_height),INTER_CUBIC);
    img.convertTo(img, CV_32FC);
    img/=255.0;
    memcpy(interpreter->typed_input_tensor<_Float32>(0), img.data, img.total() * img.elemSize());
    cout << "tensors size: " << interpreter->tensors_size() << "\n";
    cout << "nodes size: " << interpreter->nodes_size() << "\n";
    cout << "inputs: " << interpreter->inputs().size() << "\n";
    cout << "outputs: " << interpreter->outputs().size() << "\n";
    int f;
    std::vector<float> times;
    for (int i=0;i<1000;++i)
    {
    Tbegin = chrono::steady_clock::now();

    interpreter->Invoke();      // run your model

    Tend = chrono::steady_clock::now();
    f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
    times.push_back(f);
    }

    cout<<"The average processing time is: "<<std::accumulate(times.begin(),times.end(),0)/times.size()<<"ms"<<std::endl;

    cout << "Process time: " << f << " mSec" << endl;

      const float threshold = 0.001f;

    std::vector<std::pair<float, int>> top_results;

    int output = interpreter->outputs()[0];
    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    // assume output dims to be something like (1, 1, ... ,size)
    const int output_size = output_dims->data[output_dims->size - 1];
    cout << "output_size: " << output_size <<"\n";
    // Read output buffers
    // TODO(user): Insert getting data out code.
    // Note: The buffer of the output tensor with index `i` of type T can
    // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
    
    //std::vector <float> myout(output_size);
    //std::copy(interpreter->typed_output_tensor<float>(0), interpreter->typed_output_tensor<float>(0) + output_size,myout.begin());
    //for (int i=0;i<131;++i)
    //  cout<<"my out is" <<myout[i]<<endl;

    switch (interpreter->tensor(output)->type) {
        case kTfLiteFloat32:
            cout<<"Using TF Lite Float32"<<endl;
            tflite::label_image::get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                                                    5, threshold, &top_results, kTfLiteFloat32);
        break;
        case kTfLiteUInt8:
            cout<<"Using TF Lite Int8"<<endl;
            tflite::label_image::get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0), output_size,
                                                    5, threshold, &top_results, kTfLiteUInt8);
        break;
        default:
            cerr << "cannot handle output type " << interpreter->tensor(output)->type << endl;
            exit(-1);
  }

    for (const auto& result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        cout << confidence << " : " << index << "\n";
    }


  return 0;
}
