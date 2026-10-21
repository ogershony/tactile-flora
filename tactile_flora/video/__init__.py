"""Video frame reading and mosaic composition, shared by verification, eval and training.

Owns: the only implementations of frame decoding (readers.py) and mosaic layout / composition /
clip sampling (mosaic.py).
Must never: be re-implemented elsewhere; callers import from here.
"""
