#include "classify.hpp"

namespace Ort {
Classify::Classify(int numClasses,           //
                   const string& modelPath,  //
                   const optional<size_t>& gpuIdx,
                   const optional<vector<vector<int64_t>>>& inputShapes)
    : ImageRecog(numClasses, modelPath, gpuIdx, inputShapes) {}

vector<pair<int, float>> Classify::topK(
    const vector<float*>& inferenceOutput,  //
    int k,                                  //
    bool useSoftmax) const {
  auto realK = std::max(std::min(k, class_num_), 1);

  assert(inferenceOutput.size() == 1);
  float* processData = inferenceOutput[0];
  if (useSoftmax) {
    softmax(processData, class_num_);
  }

  vector<pair<int, float>> ps;
  ps.reserve(class_num_);

  for (int i = 0; i < class_num_; ++i) {
    ps.emplace_back(std::make_pair(i, processData[i]));
  }

  sort(ps.begin(), ps.end(), [](const auto& elem1, const auto& elem2) {
    return elem1.second > elem2.second;
  });

  return vector<pair<int, float>>(ps.begin(), ps.begin() + realK);
}

string Classify::topKToString(const vector<float*>& inferenceOutput,  //
                              int k,                                  //
                              bool useSoftmax) const {
  auto ps = this->topK(inferenceOutput, k, useSoftmax);

  stringstream ss;

  if (class_name_.size() == 0) {
    for (const auto& elem : ps) {
      ss << elem.first << " : " << elem.second << endl;
    }
  } else {
    for (const auto& elem : ps) {
      ss << elem.first << " : " << class_name_[elem.first] << " : "
         << elem.second << endl;
    }
  }

  return ss.str();
}
}  // namespace Ort
