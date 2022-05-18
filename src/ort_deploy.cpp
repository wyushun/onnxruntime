#include "ort_deploy.hpp"

// #include "onnxruntime/core/providers/cuda/cuda_provider_factory.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

namespace {
string toString(const ONNXTensorElementDataType dataType) {
  switch (dataType) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return "uint8_t";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return "int8_t";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return "uint16_t";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return "int16_t";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32_t";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64_t";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return "string";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "double";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return "uint32_t";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return "uint64_t";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return "complex with float32 real and imaginary components";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return "complex with float64 real and imaginary components";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "complex with float64 real and imaginary components";
    default:
      return "undefined";
  }
}
}  // namespace

namespace Ort {
class OrtDeploy::OrtDeployImpl {
 public:
  OrtDeployImpl(const string& modelPath,         //
                const optional<size_t>& gpuIdx,  //
                const optional<vector<vector<int64_t>>>& inputShapes);
  ~OrtDeployImpl();

  vector<DataOutputType> operator()(const vector<float*>& inputData);

 private:
  void initSession();
  void initModelInfo();

 private:
  string model_path_;

  Ort::Session session_;
  Ort::Env env_;
  Ort::AllocatorWithDefaultOptions ort_allocator_;

  optional<size_t> gpu_idx_;

  vector<vector<int64_t>> input_shape_;
  vector<vector<int64_t>> output_shape_;

  vector<int64_t> input_tensor_size_;
  vector<int64_t> output_tensor_size_;

  uint8_t input_num_;
  uint8_t output_num_;

  vector<char*> input_node_name_;
  vector<char*> output_node_name_;

  bool is_input_shape_provided_ = false;
};

OrtDeploy::OrtDeployImpl::OrtDeployImpl(
    const string& modelPath,         //
    const optional<size_t>& gpuIdx,  //
    const optional<vector<vector<int64_t>>>& inputShapes)
    : model_path_(modelPath),
      session_(nullptr),
      env_(nullptr),
      ort_allocator_(),
      gpu_idx_(gpuIdx),
      input_shape_(),
      output_shape_(),
      input_num_(0),
      output_num_(0),
      input_node_name_(),
      output_node_name_() {
  initSession();

  if (inputShapes.has_value()) {
    is_input_shape_provided_ = true;
    input_shape_ = inputShapes.value();
  }

  initModelInfo();
}

OrtDeploy::OrtDeployImpl::~OrtDeployImpl() {
  for (auto& elem : input_node_name_) {
    free(elem);
    elem = nullptr;
  }
  input_node_name_.clear();

  for (auto& elem : output_node_name_) {
    free(elem);
    elem = nullptr;
  }
  output_node_name_.clear();
}

void OrtDeploy::OrtDeployImpl::initSession() {
#if ENABLE_DEBUG
  env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
#else
  env_ = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "test");
#endif
  Ort::SessionOptions sessionOptions;

  // TODO: need to take care of the following line as it is related to CPU
  // consumption using openmp
  sessionOptions.SetIntraOpNumThreads(1);

  if (gpu_idx_.has_value()) {
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(
        sessionOptions, gpu_idx_.value()));
  }

  sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_ = Ort::Session(env_, model_path_.c_str(), sessionOptions);
  input_num_ = session_.GetInputCount();
  cout << "Model number of inputs: " << input_num_ << "\n";

  input_node_name_.reserve(input_num_);
  input_tensor_size_.reserve(input_num_);

  output_num_ = session_.GetOutputCount();
  cout << "Model number of outputs: " << output_num_ << "\n";

  output_node_name_.reserve(output_num_);
  output_tensor_size_.reserve(output_num_);
}

void OrtDeploy::OrtDeployImpl::initModelInfo() {
  for (int i = 0; i < input_num_; i++) {
    if (!is_input_shape_provided_) {
      Ort::TypeInfo typeInfo = session_.GetInputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

      input_shape_.emplace_back(tensorInfo.GetShape());
    }

    const auto& curInputShape = input_shape_[i];

    input_tensor_size_.emplace_back(
        std::accumulate(begin(curInputShape), end(curInputShape), 1,
                        std::multiplies<int64_t>()));

    char* inputName = session_.GetInputName(i, ort_allocator_);
    input_node_name_.emplace_back(strdup(inputName));
    ort_allocator_.Free(inputName);
  }

  {
#if ENABLE_DEBUG
    stringstream ssInputs;
    ssInputs << "Model input shapes: ";
    ssInputs << input_shape_ << endl;
    ssInputs << "Model input node names: ";
    ssInputs << input_node_name_ << endl;
    DEBUG_LOG("%s\n", ssInputs.str().c_str());
#endif
  }

  for (int i = 0; i < output_num_; ++i) {
    Ort::TypeInfo typeInfo = session_.GetOutputTypeInfo(i);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

    output_shape_.emplace_back(tensorInfo.GetShape());

    char* outputName = session_.GetOutputName(i, ort_allocator_);
    output_node_name_.emplace_back(strdup(outputName));
    ort_allocator_.Free(outputName);
  }

  {
#if ENABLE_DEBUG
    stringstream ssOutputs;
    ssOutputs << "Model output shapes: ";
    ssOutputs << output_shape_ << endl;
    ssOutputs << "Model output node names: ";
    ssOutputs << output_node_name_ << endl;
    DEBUG_LOG("%s\n", ssOutputs.str().c_str());
#endif
  }
}

vector<OrtDeploy::DataOutputType> OrtDeploy::OrtDeployImpl::operator()(
    const vector<float*>& inputData) {
  if (input_num_ != inputData.size()) {
    throw std::runtime_error("Mismatch size of input data\n");
  }

  Ort::MemoryInfo memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  vector<Ort::Value> inputTensors;
  inputTensors.reserve(input_num_);

  for (int i = 0; i < input_num_; ++i) {
    inputTensors.emplace_back(move(Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float*>(inputData[i]), input_tensor_size_[i],
        input_shape_[i].data(), input_shape_[i].size())));
  }

  auto outputTensors = session_.Run(
      Ort::RunOptions{nullptr}, input_node_name_.data(), inputTensors.data(),
      input_num_, output_node_name_.data(), output_num_);

  assert(outputTensors.size() == output_num_);
  vector<DataOutputType> outputData;
  outputData.reserve(output_num_);

  int count = 1;
  for (auto& elem : outputTensors) {
    cout << "type of input " << count++ << ": "
         << toString(elem.GetTensorTypeAndShapeInfo().GetElementType()).c_str()
         << "\n";
    outputData.emplace_back(
        make_pair(move(elem.GetTensorMutableData<float>()),
                  elem.GetTensorTypeAndShapeInfo().GetShape()));
  }

  return outputData;
}

//-----------------------------------------------------------------------------//
// OrtDeploy
//-----------------------------------------------------------------------------//

OrtDeploy::OrtDeploy(const string& modelPath,         //
                     const optional<size_t>& gpuIdx,  //
                     const optional<vector<vector<int64_t>>>& inputShapes)
    : impl_(make_unique<OrtDeployImpl>(modelPath,  //
                                       gpuIdx,     //
                                       inputShapes)) {}

OrtDeploy::~OrtDeploy() = default;

vector<OrtDeploy::DataOutputType> OrtDeploy::operator()(
    const vector<float*>& inputImgData) {
  return impl_->operator()(inputImgData);
}

}  // namespace Ort
