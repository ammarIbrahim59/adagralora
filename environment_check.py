#!/usr/bin/env python
"""Run this first on any new machine.

Reports package versions, confirms GraLoRA is importable, confirms that PEFT's
GraLoRA has no bitsandbytes backend (so 4-bit training is not an option), reports
GPU compute capability, and prints the exact --precision flag to use.

Exit code 0 means the environment is usable; 1 means something is wrong.
"""

from __future__ import annotations

import importlib
import os
import sys

# Floors below which the code is known not to work. transformers/trl are left
# unpinned above these because they move fast; peft is pinned exactly in
# requirements.txt because the mixed-k patcher uses GraloraLayer.update_layer's
# internal signature.
MIN_VERSIONS = {
    "torch": (2, 1),
    "transformers": (4, 45),
    "trl": (0, 12),
    "datasets": (2, 14),
}
PINNED_PEFT = "0.20.0"

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    GREEN = RED = YELLOW = RESET = ""


def ok(msg: str) -> None:
    print(f"  {GREEN}[ok]{RESET}   {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}[warn]{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}[FAIL]{RESET} {msg}")


def _version_tuple(v: str) -> tuple[int, ...]:
    parts = []
    for chunk in v.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_packages() -> bool:
    print("Packages")
    healthy = True

    try:
        import peft

        if peft.__version__ == PINNED_PEFT:
            ok(f"peft {peft.__version__} (pinned)")
        else:
            warn(
                f"peft {peft.__version__} but {PINNED_PEFT} is pinned. The mixed-k patcher "
                f"depends on GraloraLayer.update_layer's signature; run pytest before trusting a run."
            )
    except ImportError as exc:
        fail(f"peft not importable: {exc}")
        return False

    for name, floor in MIN_VERSIONS.items():
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "unknown")
        except ImportError as exc:
            fail(f"{name} not importable: {exc}")
            healthy = False
            continue
        if version != "unknown" and _version_tuple(version) < floor:
            fail(f"{name} {version} is below the floor {'.'.join(map(str, floor))}")
            healthy = False
        else:
            ok(f"{name} {version}")

    return healthy


def check_gralora() -> bool:
    print("\nGraLoRA availability")
    try:
        from peft import GraloraConfig  # noqa: F401
        from peft.tuners.gralora import GraloraLayer

        ok("GraloraConfig and GraloraLayer import cleanly")
    except ImportError as exc:
        fail(f"GraLoRA is not available in this peft build: {exc}")
        return False

    if not hasattr(GraloraLayer, "update_layer"):
        fail("GraloraLayer.update_layer is missing — the mixed-k patcher cannot work.")
        return False

    import inspect

    params = list(inspect.signature(GraloraLayer.update_layer).parameters)
    expected = ["self", "adapter_name", "module_name", "r", "config"]
    if params == expected:
        ok(f"update_layer signature is as expected: {params[1:]}")
    else:
        fail(f"update_layer signature changed: got {params}, expected {expected}. Pin peft=={PINNED_PEFT}.")
        return False

    import peft.tuners.gralora as gralora_pkg

    pkg_dir = os.path.dirname(gralora_pkg.__file__)
    files = sorted(f for f in os.listdir(pkg_dir) if f.endswith(".py"))
    if "bnb.py" in files:
        warn("bnb.py exists — a quantized GraLoRA backend may now be available.")
    else:
        ok("no bnb.py — GraLoRA has no bitsandbytes backend, so --load_in_4bit is impossible")

    return True


def check_gpu() -> str:
    print("\nCompute")
    try:
        import torch
    except ImportError:
        fail("torch not importable")
        return "fp32"

    if not torch.cuda.is_available():
        warn("no CUDA device visible — CPU only. Tests will run; training will not.")
        return "fp32"

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    ok(f"{name} (compute capability {major}.{minor}, {total_gb:.1f} GiB)")

    if torch.cuda.is_bf16_supported():
        ok("bf16 is supported")
        return "bf16"

    warn(
        f"bf16 is NOT supported on compute capability {major}.{minor} (this is the T4/Turing case). "
        "Use fp16 and watch closely for NaN loss."
    )
    return "fp16"


def check_constraints() -> bool:
    print("\nConstraint sweep (offline dims)")
    try:
        from adagralora.patching import KNOWN_MODELS, legal_k_values
    except ImportError as exc:
        fail(f"adagralora not importable: {exc}. Run `pip install -e .`")
        return False

    healthy = True
    for model, dims in KNOWN_MODELS.items():
        blocked = []
        for r in (16, 32, 64, 128):
            common = set((2, 4, 8))
            for in_f, out_f in dims.values():
                common &= set(legal_k_values(r, in_f, out_f))
            if not common:
                blocked.append(r)
        if blocked:
            fail(f"{model}: no k legal for every module at rank(s) {blocked}")
            healthy = False
        else:
            ok(f"{model}: k in {{2,4,8}} legal for every module at ranks 16/32/64/128")
    return healthy


def main() -> int:
    print(f"AdaGraLoRA environment check\npython {sys.version.split()[0]} at {sys.executable}\n")

    results = [check_packages(), check_gralora()]
    precision = check_gpu()
    results.append(check_constraints())

    print("\n" + "=" * 60)
    if all(results):
        print(f"{GREEN}Environment looks usable.{RESET}")
        print(f"\nUse this precision flag for training on this machine:\n\n    --precision {precision}\n")
        return 0

    print(f"{RED}Environment has problems — fix the [FAIL] lines above before training.{RESET}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
