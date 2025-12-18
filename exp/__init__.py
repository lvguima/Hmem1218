
from exp.exp_main import Exp_Main
from exp.exp_online import *
from exp.exp_solid import *

# 延迟导入函数 (避免在不需要时导入可能缺失的模块)
def _get_Exp_Proceed():
    from exp.exp_proceed import Exp_Proceed
    return Exp_Proceed

def _get_Exp_HMem():
    from exp.exp_hmem import Exp_HMem
    return Exp_HMem

# 延迟导入 - 仅在实际使用时才加载
def __getattr__(name):
    if name == 'Exp_Proceed':
        return _get_Exp_Proceed()
    if name == 'Exp_HMem':
        return _get_Exp_HMem()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")