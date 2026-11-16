from setuptools import setup
from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension

setup(
    name="rmsnorm_musa",
    ext_modules=[MUSAExtension(name="rmsnorm_musa",
                               sources=["binding.cpp", "kernel.mu"])],
    cmdclass={"build_ext": BuildExtension},
)
