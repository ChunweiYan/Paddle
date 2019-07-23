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

// This is an equivalent replacement of c++ rtti.

#include <string>
#include <type_traits>
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

class type_info {
 public:
  typedef uint32_t TypeId;

  bool operator==(const type_info& rhs) const noexcept;
  bool operator!=(const type_info& rhs) const noexcept;

  bool before(const type_info& rhs) const noexcept;

  size_t hash_code() const noexcept;
  const char* name() const noexcept;

  type_info(const type_info& rhs);
  type_info& operator=(const type_info& rhs);

 private:
  template <typename T>
  friend struct TypeDispatch;

  explicit type_info(const char*);
  TypeId id_;
};

template <typename T>
struct TypeDispatch {
  static type_info GetTypeInfo() {
    static const type_info val("unknown");
    return val;
  }
};

#define PADDLE_DECLARE_RTTI_TYPE(type)   \
  template <>                            \
  struct TypeDispatch<type> {            \
    static type_info GetTypeInfo() {     \
      static const type_info val(#type); \
      return val;                        \
    }                                    \
  }

PADDLE_DECLARE_RTTI_TYPE(bool);
PADDLE_DECLARE_RTTI_TYPE(char);
PADDLE_DECLARE_RTTI_TYPE(int);
PADDLE_DECLARE_RTTI_TYPE(std::string);
PADDLE_DECLARE_RTTI_TYPE(float);
PADDLE_DECLARE_RTTI_TYPE(double);
PADDLE_DECLARE_RTTI_TYPE(lite::Tensor);
PADDLE_DECLARE_RTTI_TYPE(std::vector<lite::Tensor>);

/*template<typename T>
struct TypeidOperator {
    //typedef
std::remove_reference<std::remove_pointer<std::remove_cv<T>::type>::type>::type
real_type;
    type_info operator()(T object) {
        return TypeDispatch<T>::GetTypeInfo();
    }
};*/

template <typename T>
inline type_info typeid(T object) {
  typedef typename std::remove_reference<typename std::remove_pointer<
      typename std::remove_cv<T>::type>::type>::type real_t;
  return TypeDispatch<real_t>::GetTypeInfo();
}

}  // namespace lite
}  // namespace paddle
