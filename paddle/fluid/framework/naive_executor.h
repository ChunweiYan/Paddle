// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/op_lite/op_lite.h"
#include "paddle/fluid/platform/device_context.h"

DECLARE_bool(global_with_gpu);

namespace paddle {
namespace framework {

/*
 * Simple, intuitive and effective. Only single thread is supported, and
 * currently designed for inference.
 */
class NaiveExecutor {
 public:
  // Either one should be set, and will be executed.
  struct Gear {
    std::unique_ptr<OperatorBase> op;
    std::unique_ptr<inference::op_lite::OpLite> lite_op;
  };

  explicit NaiveExecutor(const platform::Place& place) : place_(place) {}

  // Create child scope.
  // Create variables.
  // @with_feed_fetch_ops: whether to work with the feed and fetch operators.
  void Prepare(Scope* scope, const ProgramDesc& program_desc, int block_id,
               bool with_feed_fetch_ops);

  // Create variables before head.
  // Create parameters if persistable is ture, or create the temporary variables
  // instead.
  void CreateVariables(const ProgramDesc& desc, int block_id, bool persistable,
                       Scope* scope);

  // Run all the operators.
  void Run();

  // Get an tensor to operating directly, without the need for feed_ops.
  LoDTensor* FindTensor(const std::string& name);

  Scope* scope() { return scope_; }

 protected:
  void CreateOps(const ProgramDesc& desc, int block_id,
                 bool with_feed_fetch_ops, framework::Scope* scope = nullptr);

 private:
  const platform::Place place_;
  // Catch the required resource to avoid recreate.
  std::vector<Gear> gears_;
  Scope* scope_;
};

}  // namespace framework
}  // namespace paddle
