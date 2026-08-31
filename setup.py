"""Setup script for skordinal."""

from setuptools import setup, Extension

svor_python_extension = Extension(
    name="skordinal.classifiers._libsvor",
    sources=[
        "skordinal/classifiers/src/svor/svor_module.c",
        "skordinal/classifiers/src/svor/svor_train.c",
        "skordinal/classifiers/src/svor/svor_predict.c",
        "skordinal/classifiers/src/svor/alphas.c",
        "skordinal/classifiers/src/svor/cachelist.c",
        "skordinal/classifiers/src/svor/datalist.c",
        "skordinal/classifiers/src/svor/def_settings.c",
        "skordinal/classifiers/src/svor/loadfile.c",
        "skordinal/classifiers/src/svor/ordinal_takestep.c",
        "skordinal/classifiers/src/svor/setandfi.c",
        "skordinal/classifiers/src/svor/smo_kernel.c",
        "skordinal/classifiers/src/svor/smo_routine.c",
        "skordinal/classifiers/src/svor/smo_settings.c",
        "skordinal/classifiers/src/svor/smo_timer.c",
        "skordinal/classifiers/src/svor/svc_predict.c",
        "skordinal/classifiers/src/svor/smo_model_python.c",
        "skordinal/classifiers/src/svor/smo_loadproblem_python.c",
        "skordinal/classifiers/src/svor/smo_routine_python.c",
    ],
    extra_compile_args=["-Wno-unused-result"],
)

setup(
    ext_modules=[svor_python_extension],
)