#!/bin/bash
# Bring a fresh RunPod pytorch pod up for jumper training (run on the pod as root).
# Lessons from the first pod: mujoco needs libEGL at import; warp-lang's PyPI wheel is a CUDA 13
# build, so on a host driver below CUDA 13 the cu12 wheel from NVIDIA's GitHub release is needed.
set -e
export DEBIAN_FRONTEND=noninteractive
(apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libglvnd0 >/dev/null) || true
cd /workspace
[ -d jumper ] || git clone -q --depth 1 https://github.com/KingKongRobotics/jumper.git
cd jumper
command -v uv >/dev/null || pip install -q --break-system-packages uv
uv venv -q --seed .venv
uv pip install -q --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python .venv/bin/python -e ".[dev]"
if ! .venv/bin/python -c "import warp as wp; wp.init(); assert 'cuda:0' in [str(d) for d in wp.get_devices()]" >/dev/null 2>&1; then
  uv pip install -q --python .venv/bin/python "warp-lang @ https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl"
fi
.venv/bin/python .claude/skills/setup-env/scripts/gates.py 2>&1 | grep -E "^\[(PASS|FAIL)\] [ABCD]|devices:"
git rev-parse HEAD
