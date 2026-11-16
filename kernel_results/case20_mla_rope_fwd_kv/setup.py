from setuptools import setup
from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension

setup(
    name="mla_rope_kv_ext",
    ext_modules=[MUSAExtension(name="mla_rope_kv_ext",
                               sources=["binding.cpp", "kernel.mu"])],
    cmdclass={"build_ext": BuildExtension},
)
