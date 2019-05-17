if (NOT LITE_WITH_ARM_CPU)
    return()
endif()

cc_library(target_wrapper_arm_cpu SRCS target_wrapper.cc)
