"""Back-compat alias for `catllm`.

The canonical import name is `catllm`. `cat_llm` is also accepted so code
using the underscored convention continues to work; either form resolves to
the same module object.
"""
import importlib
import sys

_canonical = "catllm"
_real = importlib.import_module(_canonical)

sys.modules[__name__] = _real

_src_prefix = _canonical + "."
_dst_prefix = __name__ + "."
for _name in list(sys.modules):
    if _name.startswith(_src_prefix):
        sys.modules[_dst_prefix + _name[len(_src_prefix):]] = sys.modules[_name]
