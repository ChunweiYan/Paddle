if (NOT WITH_TESTING)
    return()
endif()

lite_cc_library(test_framework SRCS test_framework.cc)