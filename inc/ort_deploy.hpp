#pragma once

#include "basis.hpp"

namespace Ort {
class OrtDeploy {
 public:
  using DataOutputType = pair<float*, vector<int64_t>>;

  OrtDeploy(const string& modelPath,  //
            const optional<size_t>& gpuIdx = nullopt,
            const optional<vector<vector<int64_t>>>& inputShapes = nullopt);
  ~OrtDeploy();

  // multiple inputs, multiple outputs
  vector<DataOutputType> operator()(const vector<float*>& inputImgData);

 private:
  class OrtDeployImpl;
  unique_ptr<OrtDeployImpl> impl_;
};
}  // namespace Ort
