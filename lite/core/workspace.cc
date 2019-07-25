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

#include "lite/core/workspace.h"
paddle::lite::core::byte_t *paddle::lite::WorkSpace::Alloc(size_t size) {
  buffer_.ResetLazy(target_, cursor_ + size);
  auto* data = static_cast<core::byte_t*>(buffer_.data()) + cursor_;
  cursor_ += size;
  return data;
}
