#include "image_recog.hpp"

namespace Ort {
ImageRecog::ImageRecog(int numClasses,                  //
                       const string& modelPath,         //
                       const optional<size_t>& gpuIdx,  //
                       const optional<vector<vector<int64_t>>>& inputShapes)
    : OrtDeploy(modelPath, gpuIdx, inputShapes),
      class_num_(numClasses),
      class_name_() {
  if (numClasses <= 0) {
    throw std::runtime_error("Number of classes must be more than 0\n");
  }

  class_name_.reserve(class_num_);
  for (auto i = 0; i < class_num_; ++i) {
    class_name_.emplace_back(to_string(i));
  }
}

void ImageRecog::initClassNames(const vector<string>& classNames) {
  if (classNames.size() != static_cast<uint32_t>(class_num_)) {
    throw std::runtime_error("Mismatch number of classes\n");
  }

  class_name_ = classNames;
}

void ImageRecog::preprocess(float* dst,                 //
                            const unsigned char* src,   //
                            int64_t width,              //
                            int64_t height,             //
                            int channel,                //
                            const vector<float>& mean,  //
                            const vector<float>& variance) const {
  if (!mean.empty() && !variance.empty()) {
    assert(mean.size() == variance.size() &&
           mean.size() == static_cast<size_t>(channel));
  }

  memcpy(reinterpret_cast<void*>(dst), src, height * width * channel);

  if (!mean.empty() && !variance.empty()) {
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        for (int k = 0; k < channel; ++k) {
          dst[k * height * width + i * width + j] =
              (src[i * width * channel + j * channel + k] / 255.0 - mean[k]) /
              variance[k];
        }
      }
    }
  } else {
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        for (int k = 0; k < channel; ++k) {
          dst[k * height * width + i * width + j] =
              src[i * width * channel + j * channel + k] / 255.0;
        }
      }
    }
  }
}
}  // namespace Ort
