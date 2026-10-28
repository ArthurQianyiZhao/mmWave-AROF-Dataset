"""Load the repository's OFC MATLAB-compatible utilities by relative path."""

from __future__ import annotations
import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "OFC_2026_baseline_ML_receiver" / "utils.py"
_SPEC = importlib.util.spec_from_file_location("arof_ofc_utils", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load repository signal utilities from {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

bits_for_symbol_indices = _MODULE.bits_for_symbol_indices
matched_filter_symbols = _MODULE.matched_filter_symbols
raised_cosine_receive_filter = _MODULE.raised_cosine_receive_filter
qam_mod = _MODULE.qam_mod
qam_demod = _MODULE.qam_demod
calculate_ber = _MODULE.calculate_ber
calculate_evm = _MODULE.calculate_evm
