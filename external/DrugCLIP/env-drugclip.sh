# env_drugclip.sh

# CUDA toolkit/runtime. Prefer CUDA 12.4 when present.
if [ -d /usr/local/cuda-12.4 ]; then
  export CUDA_HOME=/usr/local/cuda-12.4
elif [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi

# Build LD_LIBRARY_PATH incrementally, only with directories that exist.
prepend_ld_library_path() {
  if [ -d "$1" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$1:"*) ;;
      *) export LD_LIBRARY_PATH="$1:${LD_LIBRARY_PATH:-}" ;;
    esac
  fi
}

# PyTorch bundled shared libraries: libc10.so, libtorch*.so, etc.
TORCH_LIB="$(python - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).parent / "lib")
PY
)"
prepend_ld_library_path "$TORCH_LIB"

# CUDA runtime libraries.
if [ -n "${CUDA_HOME:-}" ]; then
  prepend_ld_library_path "$CUDA_HOME/lib64"
fi

# Sagres-specific Spack GCC runtime. Used only if present.
SAGRES_GCC_LIB="/home/fernando.amaral/spack/linux-rocky8-zen2/gcc-14.2.0/gcc-13.3.0-jkmwxqgn4wtbri4v6nhrtgvf52nwrswa/lib64"
prepend_ld_library_path "$SAGRES_GCC_LIB"
