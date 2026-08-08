import os
import sys
from glob import glob
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

__author__ = 'Florian Hahlbohm'
__description__ = 'A refactored CUDA implementation of the 3DGS rasterizer.'

ENABLE_FASTMATH = True  # set to False to disable fast math optimizations (e.g., for debugging)
ENABLE_NVCC_LINEINFO = False  # set to True for profiling kernels with Nsight Compute (overhead is minimal)

# Detect ROCm/HIP build
IS_ROCM = hasattr(torch.version, 'hip') and torch.version.hip is not None

module_root = Path(__file__).parent.absolute()
extension_name = module_root.name
extension_root = module_root / extension_name
cuda_modules = [d.name for d in Path(extension_root).iterdir() if d.is_dir() and d.name not in ['utils', 'torch_bindings']]

# gather source files
sources = [str(extension_root / 'torch_bindings' / 'bindings.cpp')]
for module in cuda_modules:
    sources += glob(str(extension_root / module / 'src' / '**'/ '*.cpp'), recursive=True)
    sources += glob(str(extension_root / module / 'src' / '**' / '*.cu'), recursive=True)

# gather include directories
include_dirs = [str(extension_root / 'utils')]
for module in cuda_modules:
    include_dirs.append(str(extension_root / module / 'include'))

# set up compiler flags
# PyTorch 2.x uses C++20 features (concepts, requires) in headers
cxx_std = '/std:c++20' if os.name == 'nt' else '-std=c++20'
cxx_flags = [cxx_std]
if IS_ROCM:
    # HIP/ROCm flags
    nvcc_flags = ['-std=c++20']
    if ENABLE_FASTMATH:
        cxx_flags.append('-O3')
        nvcc_flags.append('-O3')
        nvcc_flags.append('-ffast-math')
else:
    # CUDA/nvcc flags
    nvcc_flags = ['-std=c++20']
    if ENABLE_FASTMATH:
        cxx_flags.append('-O3')
        nvcc_flags.append('-O3')
        nvcc_flags.append('-use_fast_math')
    if ENABLE_NVCC_LINEINFO:
        nvcc_flags.append('-lineinfo')

# Windows: c10.dll does not export the c10::ValueError(SourceLocation, string)
# inherited constructor (MSVC does not re-export inherited ctors even for C10_API
# classes). <torch/extension.h> pulls in ATen/TensorIndexing.h which triggers
# TORCH_CHECK_VALUE, generating a __declspec(dllimport) reference to that ctor
# and causing LNK2001. Redirect the missing thunk to Error(SourceLocation,string)
# which IS exported. ValueError IS-A Error with no extra data members.
extra_link_args = []
if sys.platform == "win32":
    _val_imp = (
        "__imp_??0ValueError@c10@@QEAA@USourceLocation@1@"
        "V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z"
    )
    _err_imp = (
        "__imp_??0Error@c10@@QEAA@USourceLocation@1@"
        "V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z"
    )
    extra_link_args.append(f"/ALTERNATENAME:{_val_imp}={_err_imp}")

# define the CUDA extension
extension = CUDAExtension(
    name=f'{extension_name}._C',
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args={
        'cxx': cxx_flags,
        'nvcc': nvcc_flags
    },
    extra_link_args=extra_link_args,
)

# set up the package
setup(
    name=extension_name,
    author=__author__,
    packages=[f'{extension_name}.torch_bindings'],
    ext_modules=[extension],
    description=__description__,
    cmdclass={'build_ext': BuildExtension}
)
