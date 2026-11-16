from setuptools import setup
from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension

setup(
    name="moe_unpermute_ext",
    ext_modules=[MUSAExtension(name="moe_unpermute_ext",
                               sources=["binding.cpp", "kernel.mu"])],
    cmdclass={"build_ext": BuildExtension},
)
