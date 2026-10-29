# UltraTrader — synthetic reinforcement-learning sandbox

SAC training on a hand-designed artificial market, with heuristic feature generators and a shaped reward. This is not an institutional trading system. Simulator returns, named “experts”, and reward-based ratios are not evidence of real-market alpha or a financial Sharpe ratio.

## Run

Python 3.11+; install the appropriate CPU or GPU PyTorch build for your machine, then:

```bash
python -m pip install -r requirements_ultra.txt
python -m pytest -q
ULTRA_STEPS=20000 ULTRA_SEED=123 python train_ultra.py
python eval_ultra.py
python seedeval.py
```

On PowerShell, use `$env:ULTRA_STEPS="20000"` and `$env:ULTRA_SEED="123"` before running Python. Evaluation requires a generated checkpoint. No trained checkpoint or real-market dataset is bundled.

The actual training controls are `ULTRA_STEPS` and `ULTRA_SEED`; the standalone configuration, logging and risk utilities are not all wired into that training path. MLflow tracking is not claimed as integrated. Reset seeds now cover environment, market and feature RNGs. See [AUDIT.md](AUDIT.md).
