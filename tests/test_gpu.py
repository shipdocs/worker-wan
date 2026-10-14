"""GPU architecture compatibility rule."""

import pytest

from src.gpu import arch_supported

# Arch lists as reported by torch.cuda.get_arch_list() for each build.
TORCH_25_CU124 = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
TORCH_28_CU128 = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


@pytest.mark.parametrize(
    ("capability", "arch_list", "expected"),
    [
        ((9, 0), TORCH_25_CU124, True),  # H100
        ((8, 9), TORCH_25_CU124, True),  # L40S / RTX 4090 run sm_86 code
        ((12, 0), TORCH_25_CU124, False),  # RTX 5090 / RTX PRO 6000 on torch 2.5
        ((10, 0), TORCH_25_CU124, False),  # B200
        ((12, 0), TORCH_28_CU128, True),
        ((10, 0), TORCH_28_CU128, True),
        ((8, 9), TORCH_28_CU128, True),
        ((12, 0), ["sm_90", "compute_90"], True),  # PTX JIT fallback
        ((9, 0), ["sm_90a"], True),
    ],
)
def test_arch_supported(
    capability: tuple[int, int], arch_list: list[str], expected: bool
) -> None:
    assert arch_supported(capability, arch_list) is expected
