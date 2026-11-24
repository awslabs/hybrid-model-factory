"""
Tests that HMF imports without the CUDA-only libraries that its hybrid layers need (e.g., on a Mac), with every model
family whose layers need them registered with stand-ins (see model_register.py). The tests run where none of these
libraries is installed and are skipped elsewhere.
"""

import importlib.util
import subprocess
import sys

import pytest

from hmf.model.hybrid_zoo.models.model_register import CUDA_ONLY_LIBRARIES


INSTALLED = [name for name in CUDA_ONLY_LIBRARIES if importlib.util.find_spec(name) is not None]
pytestmark = pytest.mark.skipif(bool(INSTALLED), reason=f"needs no CUDA-only library, but {INSTALLED} are installed")

# These are the modules that decision models use on a Mac, from labeling to training and serving
MODULES = (
    "hmf.model.hybrid_zoo.models.model_register",
    "hmf.priming.decision_init",
    "hmf.data.decision_label",
    "hmf.train.decision.workflow",
    "hmf.eval.decision.predictor",
    "hmf.eval.decision.server",
)


def test_modules_import_in_a_fresh_process():
    # The script runs in a fresh process, since other tests may have imported these modules already
    script = "import importlib\nfor name in %r:\n    importlib.import_module(name)\n" % (MODULES,)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_hybrid_families_get_stand_ins_that_raise_when_loaded():
    from transformers import AutoConfig, AutoModelForCausalLM

    from hmf.model.hybrid_zoo.models import model_register

    assert {"hybrid_qwen3", "hybrid_gemma4_text"} <= set(model_register._unavailable_model_types)
    config = AutoConfig.for_model("hybrid_gemma4_text")
    with pytest.raises(ImportError, match="CUDA-only libraries"):
        AutoModelForCausalLM.from_config(config)
