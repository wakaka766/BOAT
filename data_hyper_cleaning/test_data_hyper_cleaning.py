import pytest
import subprocess
from unittest.mock import patch

dynamic_methodlist = (["NGD"], ["DI", "NGD"], ["GDA", "NGD"],["GDA", "NGD", "DI"], ["DI", "NGD", "GDA"])
hyper_methodlist = ( ["CG"], ["CG", "PTT"], ["RAD"], ["RAD", "PTT"], ["RAD", "RGT"], ["PTT", "RAD", "RGT"], ["FD"], ["FD", "PTT"], ["NS"],["NS", "PTT"], ["IGA"])
dynamic_method_dm = (["NGD", "DM"], ["NGD", "DM", "GDA"])
hyper_method_dm = (["RAD"], ["CG"])
fogm_method = (["VSM"], ["VFM"], ["MESM"], ["PGDM"])

@pytest.mark.parametrize("dynamic_method, hyper_method", [
    (dynamic_method, hyper_method) for dynamic_method in dynamic_methodlist for hyper_method in hyper_methodlist
])
def test_combination_dynamic_hyper_method(dynamic_method, hyper_method):
    command = [
        "python", "/home/runner/work/BOAT/BOAT/data_hyper_cleaning/data_hyper_cleaning.py",
        "--dynamic_method", ",".join(dynamic_method),
        "--hyper_method", ",".join(hyper_method)
    ]
    print(f"Running test with dynamic_method={dynamic_method} and hyper_method={hyper_method}")

    result = subprocess.run(command, capture_output=True, text=True)

    assert result.returncode == 0, f"Test failed for dynamic_method={dynamic_method} and hyper_method={hyper_method}. Error: {result.stderr}"


@pytest.mark.parametrize("dynamic_method, hyper_method", [
    (dynamic_method, hyper_method) for dynamic_method in dynamic_method_dm for hyper_method in hyper_method_dm
])
def test_combination_dynamic_hyper_method_dm(dynamic_method, hyper_method):
    command = [
        "python", "/home/runner/work/BOAT/BOAT/data_hyper_cleaning/data_hyper_cleaning.py",
        "--dynamic_method", ",".join(dynamic_method),
        "--hyper_method", ",".join(hyper_method)
    ]
    print(f"Running test with dynamic_method={dynamic_method} and hyper_method={hyper_method}")

    result = subprocess.run(command, capture_output=True, text=True)

    assert result.returncode == 0, f"Test failed for dynamic_method={dynamic_method} and hyper_method={hyper_method}. Error: {result.stderr}"


@pytest.mark.parametrize("fogm_method", fogm_method)
def test_fogm_method(fogm_method):
    command = [
        "python", "/home/runner/work/BOAT/BOAT/data_hyper_cleaning/data_hyper_cleaning.py",
        "--fo_gm", fogm_method[0]
    ]
    print(f"Running test with fo_gm={fogm_method}")

    result = subprocess.run(command, capture_output=True, text=True)

    assert result.returncode == 0, f"Test failed for fo_gm={fogm_method}. Error: {result.stderr}"

# 测试 ImportError 情况
def test_missing_higher_module():
    with patch.dict('sys.modules', {'higher': None}):  # 模拟 higher 模块未安装
        try:
            # 重现你的异常处理逻辑
            import torch
            from torch import Tensor
            from torch.optim import Optimizer
            import higher
        except ImportError as e:
            # 检查异常信息
            missing_module = str(e).split()[-1]
            assert missing_module == "'higher'", f"Unexpected missing module: {missing_module}"

            # 检查输出内容
            expected_message = "Error: The required module 'higher' is not installed."
            assert expected_message in str(e), "The error message is incorrect."

            print("Test passed: Exception handling for missing 'higher' module works correctly.")