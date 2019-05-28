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

#include "paddle/fluid/lite/utils/any.h"
#ifdef LITE_WITH_CUDA
#include "paddle/fluid/lite/cuda/blas.h"
#include "paddle/fluid/lite/cuda/cuda_utils.h"
#endif
#ifdef LITE_WITH_X86
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device_context.h"
#endif
#include <memory>
#include <set>
#include <vector>
#include "paddle/fluid/lite/core/target_wrapper.h"

namespace paddle {
namespace lite {

struct HostContext {};

#ifdef LITE_WITH_ARM
struct ARMContext {};
#endif

#ifdef LITE_WITH_CUDA
// Only works with CUDA kernels.
struct CUDAContext {
  // overall information
  cudaStream_t exec_stream;
  cudaStream_t io_stream;

  // not thread-safe, should allocate for each thread.
  std::shared_ptr<cuda::Blas<float>> blas_fp32;

  // kernel information
  std::vector<cudaEvent_t> input_events;
  std::vector<cudaEvent_t> output_events;
};
#endif

#ifdef LITE_WITH_X86
struct X86Context {
  // overall information

  // kernel information

  // legacy info.
  std::unique_ptr<::paddle::platform::CPUDeviceContext> x86_device_context;
  std::unique_ptr<::paddle::framework::ExecutionContext> x86_execution_context;
};
#endif

// Context for running a kernel.
// Holds the necessary resource and information.
class KernelContext {
 public:
  template <typename ContextT>
  ContextT& As() {
    if (!ctx_.valid()) {
      ctx_.set<ContextT>();
    }
    return *ctx_.get_mutable<ContextT>();
  }

 private:
  Any ctx_;
};

enum class CtxFieldKind {
  kX86BlasHandler,
  kX86EigenHandler,

  kCudaIoStream,
  kCudaComputeStream,
};

class ContextFieldBase {
 public:
  virtual std::string Serialize() const = 0;
  virtual void Deserialize(const std::string& buf) = 0;
};

// X86 Field for paddle::platform::CPUDeviceContext
class X86FluidCpuDeviceContextField : public ContextFieldBase {
 public:
  // No arguments need.
  X86BlasHandler() = default;

  paddle::platform::CPUDeviceContext& data() { return *data_; }

  std::string Serialize() const override;
  void Deserialize(const std::string& buf) override;

 private:
  std::unique_ptr<paddle::platform::CPUDeviceContext> data_;
};

/// The ContextRegistry holds all the context fields in the system. It can be
/// serialized to disk to make the context analysis ahead of execution in
/// offline phase.
class ContextRegistry {
 public:
  // Create a context field using the serialized info.
  template <typename ContextFieldT>
  ContextFieldT CreateField();

  std::string Serialize() const;
  void Deserialize(const std::string&);

 private:
  std::vector<any> fields_;
};

class ContextBase {
 public:
  using fields_t = std::vector<std::pair<CtxFieldKind, std::string>>;

  template <typename ContextFieldT>
  void SetField(int id);
};

class X86Context : public ContextBase;

/// The ContextScheduler helps to assign different context for each kernel.
class ContextScheduler {
 public:
  struct FieldRepr {
    CtxFieldKind kind;
    int offset;  // offset in the field records.
  };

  ContextScheduler(ContextRegistry* registry);

  template <typename ContextT>
  void NewContext();

  template <typename ContextT>
  void NewContext(const std::vector<FieldRepr>& fields);
};

}  // namespace lite
}  // namespace paddle
