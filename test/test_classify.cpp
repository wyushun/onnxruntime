#include "classify.hpp"

#include <opencv2/opencv.hpp>

static constexpr int64_t IMG_WIDTH = 224;
static constexpr int64_t IMG_HEIGHT = 224;
static constexpr int64_t IMG_CHANNEL = 3;
static constexpr int64_t TEST_TIMES = 1000;

int main(int argc, char* argv[]) {
  if (argc != 3) {
    cerr << "Usage: [apps] [path/to/onnx/model] [path/to/image]" << endl;
    return EXIT_FAILURE;
  }

  const string ONNX_MODEL_PATH = argv[1];
  const string IMAGE_PATH = argv[2];

  Ort::Classify osh(Ort::IMAGENET_NUM_CLASSES, ONNX_MODEL_PATH, 0);
  osh.initClassNames(Ort::IMAGENET_CLASSES);

  cv::Mat img = cv::imread(IMAGE_PATH);

  if (img.empty()) {
    cerr << "Failed to read input image" << endl;
    return EXIT_FAILURE;
  }

  cv::resize(img, img, cv::Size(IMG_WIDTH, IMG_HEIGHT));
  float* dst = new float[IMG_WIDTH * IMG_HEIGHT * IMG_CHANNEL];
  osh.preprocess(dst, img.data, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL,
                 Ort::IMAGENET_MEAN, Ort::IMAGENET_STD);

  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  for (int i = 0; i < TEST_TIMES; ++i) {
    auto inferenceOutput = osh({reinterpret_cast<float*>(dst)});

    const int TOP_K = 5;
    cout << osh.topk2Str({inferenceOutput[0].first}, TOP_K) << endl;
  }
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  auto elapsedTime =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  cout << elapsedTime.count() / 1000. << "[sec]" << endl;

  delete[] dst;

  return 0;
}
