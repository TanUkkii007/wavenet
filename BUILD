package(
    default_visibility = ["//:__pkg__"],
)

py_library(
    name = "layers",
    srcs = [
        "layers/modules.py",
    ],
    srcs_version = "PY3ONLY",
    deps = [
        ":ops",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "layers_test",
    srcs = ["layers/modules_test.py"],
    deps = [
        ":layers",
    ],
    main = "layers/modules_test.py",
    srcs_version = "PY3ONLY",
    default_python_version = "PY3",
)

py_library(
    name = "ops",
    srcs = [
        "ops/convolutions.py",
        "ops/mixture_of_logistics_distribution.py",
        "ops/nonlinearity.py",
        "ops/optimizers.py",
    ],
    srcs_version = "PY3ONLY",
    visibility = ["//visibility:public"],
)

py_test(
    name = "ops_test",
    srcs = ["ops/convolutions_test.py"],
    deps = [
        ":ops",
    ],
    main = "ops/convolutions_test.py",
    srcs_version = "PY3ONLY",
    default_python_version = "PY3",
)

py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    srcs_version = "PY3ONLY",
    default_python_version = "PY3",
    deps = [
        ":layers",
    ],
)

py_binary(
    name = "predict",
    srcs = [
        "predict.py",
    ],
    srcs_version = "PY3ONLY",
    default_python_version = "PY3",
    deps = [
        ":layers",
    ],
)