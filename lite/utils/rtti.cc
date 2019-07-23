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

/*
 * This file implements an lightweight alternative for glog, which is more
 * friendly for mobile.
 */

#include "lite/utils/rtti.h"
// https://github.com/google/styleguide/commit/52aa2e05491304ea047e29a873c4cd9dc2054a63
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace lite {

// singleton manager for rtti type system
struct TypeManager {
  uint32_t rtti_counter{0};

  std::mutex mutex;
  std::unordered_map<std::string, uint32_t> key2index;
  std::vector<std::string> index2key;

  // get singleton of the
  static TypeManager* Global() {
    static TypeManager inst;
    return &inst;
  }
};

type_info(const type_info& rhs) { id_ = rhs.id_; }

type_info& operator=(const type_info& rhs) { id_ = rhs.id_; }

bool operator==(const type_info& rhs) { return hash_code() == rhs.hash_code(); }

bool operator!=(const type_info& rhs) { return hash_code() != rhs.hash_code(); }

bool before(const type_info& rhs) { return hash_code() < rhs.hash_code(); }

size_t hash_code() { return id_; }

const char* name() { return index2key[id_].c_str(); }

type_info(const char* name) {
  auto* mgr = TypeManager::Global();
  std::lock_guard<std::mutex> lock(mgr->mutex);
  std::string sname = name;
  auto it = t->key2index.find(sname);
  if (it == t->key2index.end()) {
    index2key.push_back(sname);
    key2index[sname] = rtti_counter++;
  }
  id_ = key2index[sname];
}

}  // namespace lite
}  // namespace paddle
