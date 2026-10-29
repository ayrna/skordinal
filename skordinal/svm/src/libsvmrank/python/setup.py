from distutils.core import setup, Extension

svm_python_extension = Extension(
    name="svm",
    language="c++",
    sources=[
        "svm-module.cpp",
        "svm-train.cpp",
        "svm-predict.cpp",
        "svm-model-python.cpp",
        "svm.cpp",
    ],
    extra_compile_args=["-lefence", "-Wno-unused-result"],
)

setup(name="libsvmRank", ext_modules=[svm_python_extension])
