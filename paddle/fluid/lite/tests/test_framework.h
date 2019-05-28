#pragma once
/*
* This file implements lite::TestFramework, a framework for executing the lite
* and legacy fluid framework and compare the performance.
*/

#include <gtest/gtest.h>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/framework.pb.h"
#include "paddle/fluid/lite/core/program.h"
#include "paddle/fluid/lite/model_parser/pb/op_desc.h"

namespace paddle {
namespace lite {
namespace tests {

/*
 * FluidTester is used to execute several operators and get the output.
 */
class FluidTester {
 public:
  FluidTester(const std::vector<framework::proto::OpDesc>& ops) {
    for (auto& desc : ops) {
      ops_.emplace_back(framework::OpRegistry::CreateOp(desc));
    }
    scope_.reset(new framework::Scope);
  }

  void Run() {
    for (auto& op : ops_) {
      op->Run(*scope_, platform::CPUPlace());
    }
  }

  framework::LoDTensor* DeclTensor(const std::string& key) {
    return scope_->Var(key)->GetMutable<framework::LoDTensor>();
  }

  const framework::LoDTensor& GetTensor(const std::string& key) {
    return scope_->FindVar(key)->Get<framework::LoDTensor>();
  }

 private:
  std::unique_ptr<framework::Scope> scope_;
  std::vector<std::unique_ptr<framework::OperatorBase>> ops_;
};

class LiteTester {
 public:
  LiteTester(const std::vector<framework::proto::OpDesc>& ops,
             const std::vector<lite::Place>& valid_places) {
    // fake a program desc
    framework::ProgramDesc program_desc;
    auto* main_block = program_desc.MutableBlock(0);

    // Leave the var list empty, all the vars in the scope should be created
    // externally.
    lite::Program program(ops, *scope_, valid_places);
  }

  lite::Tensor* DeclTensor(const std::string& key) {
    return scope_->Var(key)->GetMutable<lite::Tensor>();
  }
  const lite::Tensor& GetTensor(const std::string& key) {
    return scope_->FindVar(key)->Get<lite::Tensor>();
  }

  void Run();

 private:
  std::unique_ptr<lite::Scope> scope_;
};

/*
 * The OpsTesterForLiteAndFluid executes a list of operators and compare the
 * output in both scenerios.
 */
class OpsTesterForLiteAndFluid {
 public:
  OpsTesterForLiteAndFluid(const std::vector<framework::proto::OpDesc> ops) {
    fluid_tester_.reset(new FluidTester(ops));
    lite_tester_.reset(new LiteTester(ops));
  }

  template <typename T>
  void SetTensor(const std::string& key, const std::vector<int64_t>& shape,
                 void* data) {
    // set tensor in fluid
    auto* fluid_tensor = fluid_tester_->DeclTensor(key);
    fluid_tensor->Resize(framework::make_ddim(shape));
    ;

    size_t buffer_size =
        framework::product(framework::make_ddim(shape)) * sizeof(T);
    std::copy_n(data, buffer_size,
                fluid_tensor->mutable_data<T>(platform::CPUPlace()));

    // set tensor in lite
    auto* lite_tensor = lite_tester_->DeclTensor(key);
    lite_tensor->Resize(shape);
    std::copy_n(data, buffer_size, lite_tensor->mutable_data<T>());
  }

  // Execute the operators and compare the output.
  void Test() {
    fluid_tester_->Run();
    lite_tester_->Run();
  }
  // Compare the two output tensor from lite and fluid.
  template <typename T>
  void CompareTensor(const std::string& key) {
    const auto& fluid_tensor = fluid_tester_->GetTensor(key);
    const auto& lite_tensor = lite_tester_->GetTensor(key);

    // check shape
    auto shape0 = framework::vectorize(fluid_tensor.dims());
    auto shape1 = lite_tensor->dims();
    ASSERT_EQ(shape0.size(), shape1.size());
    for (int i = 0; i < shape0.size(); i++) {
      EXPECT_EQ(shape0[i], shape1[i]);
    }

    // check data
    const T* data0 = fluid_tensor.data<T>();
    const T* data1 = lite_tensor->data<T>();

    size_t num_elem = framework::product(fluid_tensor.dims());
    for (size_t i = 0; i < num_elem; i++) {
      EXPECT_NEAR(data0[i], data1[i], 1e-6);
    }
  }

 private:
  std::unique_ptr<FluidTester> fluid_tester_;
  std::unique_ptr<LiteTester> lite_tester_;
};

}  // namespace tests
}  // namespace lite
}  // namespace paddle
