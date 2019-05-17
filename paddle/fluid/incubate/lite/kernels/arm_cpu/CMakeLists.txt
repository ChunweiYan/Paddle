message(STATUS "compile with lite ARM CPU kernels")

cc_library(relu_compute_arm_cpu SRCS relu_compute.cc DEPS ${lite_kernel_deps})

set(arm_cpu_kernels
    relu_compute_arm_cpu
    )

set(arm_cpu_kernels "${arm_cpu_kernels}" CACHE INTERNAL "arm cpu kernels")
