# Shared environment for the GPU study artifacts (kastner-ml, RTX A6000 sm_86)
export UV_CACHE_DIR=/var/tmp/lg-env/uv-cache
export UV_PYTHON_INSTALL_DIR=/var/tmp/lg-env/pythons
export VENV=/var/tmp/lg-env/venv
export PATH="$HOME/.nix-profile/bin:$PATH"
if [ -d "$VENV" ]; then . "$VENV/bin/activate"; fi
export TORCHINDUCTOR_CACHE_DIR=/var/tmp/lg-env/inductor-cache
export TRITON_CACHE_DIR=/var/tmp/lg-env/triton-cache
export HF_HOME=/var/tmp/lg-env/hf
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# NixOS: the NVIDIA userspace driver is not on the default loader path.
# Without this, torch reports "Found no NVIDIA driver on your system".
export LD_LIBRARY_PATH="/run/opengl-driver/lib:${LD_LIBRARY_PATH}"

# NixOS: Triton discovers libcuda by shelling out to /sbin/ldconfig, which
# does not exist here. TRITON_LIBCUDA_PATH is the supported override.
export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
# Triton compiles its launcher stub with a host C compiler; NixOS has none on
# the default PATH, so point CC at the nix-profile gcc.
export CC="$HOME/.nix-profile/bin/gcc"
export CXX="$HOME/.nix-profile/bin/g++"
