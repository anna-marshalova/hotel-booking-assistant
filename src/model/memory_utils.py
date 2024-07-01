import gc
import sys
import traceback

import torch
from IPython import get_ipython


def clean_ipython_hist():
    # Code in this function mainly copied from IPython source
    if "get_ipython" not in globals():
        return
    ip = get_ipython()
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc):
        user_ns.pop("_i" + repr(n), None)
    user_ns.update(dict(_i="", _ii="", _iii=""))
    hm = ip.history_manager
    hm.input_hist_parsed[:] = [""] * pc
    hm.input_hist_raw[:] = [""] * pc
    hm._i = hm._ii = hm._iii = hm._i00 = ""


def clean_tb():
    # h/t Piotr Czapla
    if hasattr(sys, "last_traceback"):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, "last_traceback")
    if hasattr(sys, "last_type"):
        delattr(sys, "last_type")
    if hasattr(sys, "last_value"):
        delattr(sys, "last_value")


def cleanup(n: int = 1):
    for _ in range(n):
        clean_tb()
        clean_ipython_hist()
        gc.collect()
    torch.cuda.empty_cache()
