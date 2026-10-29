# 运行与真实性审查 — 2026-10-29

## 补齐训练依赖并修复随机种子

reset(seed)现在同时控制环境、市场模拟和专家特征的随机数生成器。补齐训练所需依赖及进度条依赖；删除过期任务清单和本机权限配置；README以真正读取的ULTRA_STEPS/ULTRA_SEED为准。

## 本次执行证据

7项测试通过；CPU执行120步SAC短训练，成功保存checkpoint并推理一次。

```text
python -m pytest -q
ULTRA_STEPS=120 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_ultra.py
```

Python检查使用Linux/Python 3.12。测试结果是本次本地执行记录，不代表GitHub Actions已通过，也不是独立认证。

## 仍未验证的范围

120步只证明训练入口、梯度更新、保存和推理可运行，不代表收敛。市场与专家输出均为模拟或启发式；真实交易效果未验证。

删除仅针对已确认失效、重复说明或可再生成产物。源码、必要资产和不同版本保留；更改可从Git提交历史恢复。
