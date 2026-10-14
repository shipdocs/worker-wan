"""Check that the installed PyTorch has kernels for the GPU (and for the GPUs we deploy on)."""

import os

# Ampere, Ada, Hopper, Blackwell (B200 and RTX 50xx / RTX PRO 6000).
# Override with REQUIRED_CUDA_ARCHS="" (e.g. for a CPU-only dev environment).
REQUIRED_CUDA_ARCHS = os.environ.get(
    "REQUIRED_CUDA_ARCHS", "sm_80 sm_86 sm_89 sm_90 sm_100 sm_120"
).split()


def arch_supported(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """Whether a torch build for ``arch_list`` can run on a GPU of ``capability``.

    Follows CUDA's compatibility rules: ``sm_XY`` binaries run on any ``X.Z`` GPU
    with ``Z >= Y`` (so sm_86 covers Ada sm_89), and ``compute_XY`` PTX can be
    JIT-compiled for any newer GPU. Blackwell (sm_120) needs sm_120 or PTX.
    """
    for entry in arch_list:
        kind, _, digits = entry.partition("_")
        digits = digits.rstrip("abcdefghijklmnopqrstuvwxyz")  # e.g. sm_90a
        if not digits.isdigit() or len(digits) < 2:
            continue
        built = (int(digits[:-1]), int(digits[-1]))
        if kind == "sm" and built[0] == capability[0] and built[1] <= capability[1]:
            return True
        if kind == "compute" and built <= capability:
            return True
    return False


def check_cuda_compatibility() -> None:
    """Fail fast if the installed PyTorch has no kernels for this GPU.

    Otherwise an unsupported GPU only logs a warning at startup and every job then
    fails with "no kernel image is available for execution on the device".
    """
    import torch

    if not torch.cuda.is_available():
        msg = "CUDA is not available; this worker needs an NVIDIA GPU"
        raise RuntimeError(msg)

    capability = torch.cuda.get_device_capability()
    arch = "sm_{}{}".format(*capability)
    supported = torch.cuda.get_arch_list()
    print(
        f"torch {torch.__version__} (CUDA {torch.version.cuda}), "
        f"GPU {torch.cuda.get_device_name()} ({arch}), built for {supported}",
        flush=True,
    )
    if not arch_supported(capability, supported):
        msg = (
            f"GPU architecture {arch} is not supported by torch {torch.__version__} "
            f"(built for {supported}). Use a different GPU type for this endpoint."
        )
        raise RuntimeError(msg)


def check_torch_build() -> None:
    """Build-time check (no GPU needed) that torch has kernels for REQUIRED_CUDA_ARCHS."""
    import torch

    # get_arch_list() returns [] without a GPU; the compiled flags are always there.
    flags = torch._C._cuda_getArchFlags() if torch.version.cuda else ""
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, archs: {flags}")
    missing = [
        arch
        for arch in REQUIRED_CUDA_ARCHS
        if not arch_supported((int(arch[3:-1]), int(arch[-1])), flags.split())
    ]
    if missing:
        msg = f"torch {torch.__version__} is not built for {missing}"
        raise SystemExit(msg)
