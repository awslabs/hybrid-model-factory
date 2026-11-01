"""
Wrappers for fla's private ``chunk_fwd_h`` and ``chunk_bwd_dh``
(``fla.ops.common.chunk_h``) that take natural-log gates on every fla version.

GKA's Chebyshev ops call these functions directly. From fla 0.5.1 they decay their
states with ``exp2``, so they expect gates in log2 units, which fla's own wrappers
(e.g. ``chunk_simple_gla``) get by scaling the gates by ``RCP_LN2 = 1/ln(2)``.
Passed natural-log gates, every decay would be ``2^g`` instead of ``e^g``. Earlier
versions use ``exp`` and take natural-log gates as they are.

The convention is detected by checking whether the installed ``chunk_h`` source uses
``exp2``, rather than from the version number. Only the gates passed to these two
functions are rescaled: GKA's own Triton kernels use natural ``exp``, so its gate
tensors stay in natural-log units.
"""

import inspect
import math

import fla.ops.common.chunk_h as _chunk_h
from fla.ops.common.chunk_h import chunk_bwd_dh as _chunk_bwd_dh
from fla.ops.common.chunk_h import chunk_fwd_h as _chunk_fwd_h

_FLA_USES_EXP2 = "exp2" in inspect.getsource(_chunk_h)

if _FLA_USES_EXP2:
    # Import the canonical constant from fla (present in every fla that uses
    # exp2). Hard-fail if missing: silently not scaling would corrupt h_kk.
    from fla.ops.utils.constant import RCP_LN2
else:
    # exp-kernel fla (e.g. 0.4.1): gates pass through unscaled. _to_log2 never
    # applies this value; 1/ln(2) is defined only so the module is uniform.
    RCP_LN2 = 1.0 / math.log(2)


def _to_log2(t):
    """Rescale a natural-log gate tensor to log2 space iff the installed fla's
    chunk_h kernels expect it (exp2 convention). No-op on exp-kernel fla."""
    if t is None or not _FLA_USES_EXP2:
        return t
    return t * RCP_LN2


def chunk_fwd_h(*, g=None, gk=None, gv=None, **kwargs):
    """``fla.ops.common.chunk_h.chunk_fwd_h`` taking natural-log gates on every
    fla version. All arguments must be passed by keyword."""
    return _chunk_fwd_h(g=_to_log2(g), gk=_to_log2(gk), gv=_to_log2(gv), **kwargs)


def chunk_bwd_dh(*, g=None, gk=None, gv=None, **kwargs):
    """``fla.ops.common.chunk_h.chunk_bwd_dh`` taking natural-log gates on every
    fla version. All arguments must be passed by keyword.

    Returns dh/dh0 (state gradients), which need no unscaling: with the inputs
    rescaled, the kernel's decay factors match the natural-log convention.
    """
    return _chunk_bwd_dh(g=_to_log2(g), gk=_to_log2(gk), gv=_to_log2(gv), **kwargs)
