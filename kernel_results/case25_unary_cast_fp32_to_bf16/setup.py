from setuptools import setup
from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension

setup(
    name="_cast_fp32_to_bf16_rne_musa",
    ext_modules=[MUSAExtension(name="_cast_fp32_to_bf16_rne_musa",
                               sources=["binding.cpp", "kernel.mu"])],
    cmdclass={"build_ext": BuildExtension},
)
