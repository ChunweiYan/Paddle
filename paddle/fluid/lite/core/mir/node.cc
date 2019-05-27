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

#include "paddle/fluid/lite/core/mir/node.h"

namespace paddle {
namespace lite {
namespace mir {

std::ostream &operator<<(std::ostream &os, Node &other) {
  os << static_cast<int>(other.role_) << " ";
  if (!other.IsRoleSet()) {
    os << "Unk role node";
  }
  if (other.IsArg()) {
    auto &arg = other.AsArg();
    os << "Argument " << arg.name;
  }
  if (other.IsStmt()) {
    auto &arg = other.AsStmt();
    os << "Statement " << arg.op_type;
  }
  return os;
}

Node::Stmt &Node::AsStmt() {
  if (role_ != Role::kUnk) {
    CHECK(role_ == Role::kStmt);
    return *stmt_;
  }
  role_ = Role::kStmt;
  stmt_.reset(new Stmt);
  return *stmt_;
}

Node::Arg &Node::AsArg() {
  if (role_ != Role::kUnk) {
    CHECK(role_ == Role::kArg);
    return *arg_;
  }
  role_ = Role::kArg;
  arg_.reset(new Arg);
  return *arg_;
}

Node::Stmt &Node::AsStmt(const std::string &op_type,
                         std::vector<std::unique_ptr<KernelBase>> &&kernels,
                         const std::shared_ptr<OpLite> &op) {
  auto &x = AsStmt();
  x.op_type = op_type;
  x.op = op;
  x.valid_kernels = std::move(kernels);
  return x;
}

Node::Arg &Node::AsArg(const std::string &name) {
  auto &x = AsArg();
  x.name = name;
  return x;
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
