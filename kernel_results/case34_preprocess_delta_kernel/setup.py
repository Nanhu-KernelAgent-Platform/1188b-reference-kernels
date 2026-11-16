from setuptools import setup
from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension

setup(
    name="preprocess_delta_ext",
    ext_modules=[MUSAExtension(name="preprocess_delta_ext",
                               sources=["binding.cpp", "kernel.mu"])],
    cmdclass={"build_ext": BuildExtension},
)
