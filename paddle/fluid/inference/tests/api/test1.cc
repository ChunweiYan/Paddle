#include <dirent.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <chrono>
#include <fstream>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
using paddle::contrib::AnalysisConfig;
using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

DEFINE_string(dirname, "./", "Directory of the inference model.");
DEFINE_int32(batch, 1, "Directory of the inference model.");

void convert_output(const std::vector<paddle::PaddleTensor> &tensors,
                    std::vector<std::vector<float>> &datas,
                    std::vector<std::vector<int>> &shapes) {
  // use reference to avoid double free
  for (auto &t : tensors) {
    shapes.push_back(t.shape);
    const size_t num_elements = t.data.length() / sizeof(float);
    float *tdata = static_cast<float *>(t.data.data());
    std::vector<float> data(num_elements, 0);
    std::copy(tdata, tdata + num_elements, data.data());
    datas.push_back(data);
  }
}

std::string fluid_predict(paddle::PaddlePredictor *pd_predictor,
                          std::string &file_c) {
  int height = 400;
  int width = 400;
  std::vector<paddle::PaddleTensor> input_tensors;
  std::vector<paddle::PaddleTensor> output_tensors;
  // parent_idx
  // image tensor
  //
  std::stringstream ss;
  ss.str(file_c);

  std::string file_name;
  int b_;

  ss >> file_name;
  ss >> b_;
  ss >> height;
  ss >> width;

  paddle::PaddleTensor image_tensor;
  std::vector<int> image_shape;
  image_shape.push_back(FLAGS_batch);
  image_shape.push_back(1);
  image_shape.push_back(height);
  image_shape.push_back(width);
  std::vector<float> image_data;

  std::cout << height << " " << width << std::endl;
  float temp_v;

  for (int b = 0; b < FLAGS_batch; b++) {
    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
        ss >> temp_v;
        // image_data.push_back((float)());
        image_data.push_back(static_cast<float>(temp_v));
      }
    }
  }

  std::cerr << "image size is " << image_data.size() << std::endl;
  image_tensor.shape = image_shape;
  image_tensor.dtype = paddle::PaddleDType::FLOAT32;

  image_tensor.data.Resize(sizeof(float) * height * width * FLAGS_batch);
  std::copy(image_data.begin(), image_data.end(),
            static_cast<float *>(image_tensor.data.data()));
  image_tensor.name = "pixel";

  paddle::PaddleTensor init_ids_tensor;
  std::vector<int> ids_shape;
  ids_shape.push_back(FLAGS_batch);
  ids_shape.push_back(1);
  std::vector<int64_t> init_ids;
  for (int i = 0; i < FLAGS_batch; i++) init_ids.push_back(0);
  init_ids_tensor.shape = ids_shape;
  init_ids_tensor.dtype = paddle::PaddleDType::INT64;
  // init_ids_tensor.data = init_ids;
  init_ids_tensor.data.Resize(sizeof(int64_t) * FLAGS_batch);
  init_ids_tensor.name = "init_ids";
  std::copy(init_ids.begin(), init_ids.end(),
            static_cast<int64_t *>(init_ids_tensor.data.data()));
  std::vector<size_t> lod_1;
  for (int i = 0; i <= FLAGS_batch; i++) {
    lod_1.push_back(i);
  }

  std::vector<size_t> lod_2;
  for (int i = 0; i <= FLAGS_batch; i++) {
    lod_2.push_back(i);
  }

  std::vector<std::vector<size_t>> lod;
  lod.push_back(lod_1);
  lod.push_back(lod_2);
  init_ids_tensor.lod = lod;

  // init scores
  paddle::PaddleTensor init_scores_tensor;
  std::vector<int> scores_shape;
  scores_shape.push_back(FLAGS_batch);
  scores_shape.push_back(1);
  std::vector<float> init_scores;
  for (int i = 0; i < FLAGS_batch; i++) init_scores.push_back(0.0);
  init_scores_tensor.shape = scores_shape;
  init_scores_tensor.dtype = paddle::PaddleDType::FLOAT32;
  // init_scores_tensor.data = init_scores;
  init_scores_tensor.data.Resize(sizeof(float) * FLAGS_batch);
  std::copy(init_scores.begin(), init_scores.end(),
            static_cast<float *>(init_scores_tensor.data.data()));
  init_scores_tensor.name = "init_scores";
  init_scores_tensor.lod = lod;

  input_tensors.push_back(image_tensor);
  input_tensors.push_back(init_ids_tensor);
  input_tensors.push_back(init_scores_tensor);
  std::cerr << "before prediction \n";
  pd_predictor->Run(input_tensors, &output_tensors);
  auto time1 = time();
  std::cerr << "start new prediction \n";
  for (int i = 0; i < 100; i++) {
    pd_predictor->Run(input_tensors, &output_tensors);
  }

  auto time2 = time();
  std::cout << "batch: " << FLAGS_batch
            << " predict cost: " << time_diff(time1, time2) / 100.0 << "ms"
            << std::endl;

  // platform::EnableProfiler(platform::ProfilerState::kAll);
  pd_predictor->Run(input_tensors, &output_tensors);
  // platform::DisableProfiler(platform::EventSortingKey::kTotal,
  // "./dinge_ocr_timeline.file");

  std::vector<std::vector<float>> output_data;
  std::vector<std::vector<int>> output_shapes;
  convert_output(output_tensors, output_data, output_shapes);

  std::string plate_str = "";
  for (float k : output_data[0]) {
    //       std::cerr << "text\t" << k << std::endl;
    if (k == 0 || k == 1 || k == 2) {
      continue;
    }
  }

  std::cout << "real_size: " << output_data[1].size() << std::endl;
  for (float k : output_data[1]) {
    //	    std::cerr << "scores\t" << k << std::endl;
    if (k) {
    }
    continue;
  }
  return plate_str;
}

void PrepareTRTConfig(AnalysisConfig *config, int batch_size) {
  std::string model_dir = "/home/chunwei/project2/models/dinge_fluid/dinge";
  config->prog_file = model_dir + "/model";
  config->param_file = model_dir + "/params";
  config->use_gpu = false;
  config->device = 0;
  config->specify_input_name = true;
  config->pass_builder()->DeletePass("subblock_to_graph_pass");
  config->pass_builder()->InsertPass(
      config->pass_builder()->AllPasses().size() - 2, "subblock_to_graph_pass");
  config->pass_builder()->TurnOnDebug();
}

int run() {
  // 1. init image recognition model
  AnalysisConfig config(false);
  PrepareTRTConfig(&config, 1);
  // NativeConfig config;
  // PrepareNativeConfig(&config, 1);

  // auto fluid_predictor = CreatePaddlePredictor<NativeConfig>(config);
  auto fluid_predictor = CreatePaddlePredictor(config);

  double total_time_cost = 0.0f;
  int total_number = 1;

  timeval start_time;
  gettimeofday(&start_time, NULL);
  std::fstream out_file("./save.txt");
  std::string ana_line;
  int index = 0;
  while (std::getline(out_file, ana_line) && index < 1) {
    std::cout << "index: " << index << std::endl;
    std::string predict_str = fluid_predict(fluid_predictor.get(), ana_line);
    index += 1;
  }
  timeval end_time;
  gettimeofday(&end_time, NULL);
  double time_cost = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                     (end_time.tv_usec - start_time.tv_usec) / 1000.0;
  std::cout << "predict time is [" << time_cost << "]" << std::endl;
  total_time_cost += time_cost;
  total_number += 1;
  std::cout << "avg time cost is [" << total_time_cost / total_number << "]"
            << std::endl;

  return 0;
}
}  // namespace paddle

TEST(test, test) { paddle::run(); }
