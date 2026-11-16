import os
from setuptools import setup
from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension

os.environ.setdefault("TORCH_MUSA_ARCH_LIST", "31")
setup(
    name="rotary_bwd_kv_musa",
    ext_modules=[MUSAExtension(
        name="rotary_bwd_kv_musa",
        sources=["src/binding.cpp", "src/kernel.mu"],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"], "mcc": ["-O3"]},
    )],
    cmdclass={"build_ext": BuildExtension},
)
