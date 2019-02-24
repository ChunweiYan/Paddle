// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <memory>
#include "target_wrapper.h"

namespace paddle {
namespace lite {

enum class OpTarget {
  kHost = 0,
  kX86,
  kCUDA,
  kARM,
};

template <TargetType Target>
class Context {
 public:
  using target_wrapper_t = TargetWrapper<Target>;
  using stream_t = typename TargetWrapper<Target>::stream_t;

  Context() = default;
  Context(int device_id, stream_t compute_stream, stream_t data_stream)
      : device_id_(device_id),
        compute_stream_(compute_stream),
        data_stream_(data_stream) {}

  void SetDeviceId(int device_id) { device_id_ = device_id; }
  void SetComputeStream(stream_t x) { compute_stream_ = x; }
  void SetDataStream(stream_t x) { data_stream_ = x; }

  int device_id() const { return device_id_; }
  stream_t compute_stream() const { return compute_stream_; }
  stream_t data_stream() const { return data_stream_; }

 private:
  int device_id_;
  stream_t compute_stream_;
  stream_t data_stream_;
};

class OpContext final {
 public:
  template <TargetType Target>
  using target_ptr_t = std::unique_ptr<Context<Target>>;

  // @param target valid target.
  explicit OpContext(OpTarget target)
      : targets_(std::vector<OpTarget>({target})) {}
  // @param target valid target.
  explicit OpContext(const std::vector<OpTarget>& target) : targets_(target) {}

  const std::vector<OpTarget>& target() const { return targets_; }

  template <TargetType Target>
  target_ptr_t<Target> CreateContext() {
    return target_ptr_t<Target>(new Context<Target>);
  }

 private:
  std::vector<OpTarget> targets_;
};

}  // namespace lite
}  // namespace paddle
