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

#include <glog/logging.h>
#include <boost/variant.hpp>
#include <map>
#include <string>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/lite/context.h"
#include "paddle/fluid/lite/target_wrapper.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {

// Light-weight kernel implementation.
// The OpKernel is designed to implement the specific algorithm on a target
// device.
template <TargetType Target, PrecisionType Precision>
class OpKernel {
 public:
  using context_t = Context<Target>;
  using context_ptr_t = std::unique_ptr<context_t>;

  OpKernel() = default;

  void SetContext(context_ptr_t&& ctx) { context_ = std::move(ctx); }

  void SetParam(any param) { param_ = param; }

  template <typename Param>
  Param& param() const {
    return *any_cast<Param>(&param_);
  }

  virtual void Run() { CHECK(false) << "Not Implemented"; }

  virtual ~OpKernel() = default;

 protected:
  context_ptr_t context_;
  mutable any param_;
};

}  // namespace lite
}  // namespace paddle
