#pragma once

#include "basis.hpp"
#include "ort_deploy.hpp"

namespace Ort {
class ImageRecog : public OrtDeploy {
 public:
  ImageRecog(int numClasses,                            //
             const string& modelPath,                   //
             const optional<size_t>& gpuIdx = nullopt,  //
             const optional<vector<vector<int64_t>>>& inputShapes = nullopt);

  ~ImageRecog() = default;

  void initClassNames(const vector<string>& classNames);

  virtual void preprocess(float* dst,                      //
                          const unsigned char* src,        //
                          int64_t width,                   //
                          int64_t height,                  //
                          int channel,                     //
                          const vector<float>& mean = {},  //
                          const vector<float>& variance = {}) const;

  int numClasses() const { return class_num_; }

  const vector<string>& classNames() const { return class_name_; }

 protected:
  int class_num_;
  vector<string> class_name_;
};
}  // namespace Ort
