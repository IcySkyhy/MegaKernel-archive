import triton


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx90a"


def flatten(S):
    """
    Flatten a recursive list or tuple
    """
    if isinstance(S, (list, tuple)) and len(S) == 0:
        return list(S)
    if isinstance(S[0], (list, tuple)):
        return flatten(S[0]) + flatten(S[1:])
    return list(S[:1]) + flatten(S[1:])
