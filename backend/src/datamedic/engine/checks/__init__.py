"""Diagnostic checks.

``spec`` holds lightweight declarations (stdlib only, importable by the API). ``framework``
runs checks over a frame (worker only). Implementations are referenced from specs by a
"module:function" path and imported lazily.
"""
