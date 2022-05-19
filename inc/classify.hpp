#pragma once

#include "image_recog.hpp"

namespace Ort {
class Classify : public ImageRecog {
 public:
  Classify(int numClasses,                            //
           const string& modelPath,                   //
           const optional<size_t>& gpuIdx = nullopt,  //
           const optional<vector<vector<int64_t>>>& inputShapes = nullopt);

  ~Classify() = default;

  vector<pair<int, float>> topk(const vector<float*>& inferenceOutput,  //
                                int k = 1,                              //
                                bool useSoftmax = true) const;

  string topk2Str(const vector<float*>& inferenceOutput,  //
                  int k = 1,                              //
                  bool useSoftmax = true) const;
};
}  // namespace Ort
