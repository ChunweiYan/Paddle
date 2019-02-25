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
#include "context.h"
#include "op_kernel.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace lite {

using any_t = boost::variant<int, float, framework::Variable *>;
using anys_t = std::map<std::string, any_t>;

// For registry factory.
struct Registry {
  void Touch() {}
};

/**
 * The base class of an light-weight operators, currently just used in inference
 * to eliminate overhead of some operations in current framework.
 *
 * The Operator are designed as follows:
 * - it can has some members to hold the argument addresses,
 * - it should act just like a function call, no more logic should included.
 */
class OpLite : public Registry {
 public:
  enum class KernelStrategy {
    // Return the user specified one.
    kStatic = 0,
    // Specify the expected kernel externally.
    kSpecified,
    // Run each kernel to evaluate and get the best kernel.
    kRuntime,
  };

  OpLite() {}
  OpLite(std::unique_ptr<OpContext> &&x) : op_context_(std::move(x)) {}

  virtual bool CheckShape() const { return true; }
  virtual bool InferShape() const { return true; }
  virtual bool Run() = 0;
  virtual bool Build(const framework::OpDesc &opdesc,
                     framework::Scope *scope) = 0;
  virtual std::string DebugString() const = 0;

  virtual void StaticPickKernel(const std::vector<OpTarget> &valid_targets) = 0;

  void PickBestKernel(const std::vector<OpTarget> &valid_places,
                      KernelStrategy kernel_strategy = KernelStrategy::kStatic);

  // Create all the kernels for the valid targets.
  void CreateKernels();

  virtual ~OpLite() = default;

 protected:
  std::unique_ptr<OpContext> op_context_;
};

}  // namespace lite
}  // namespace paddle
