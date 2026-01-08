# final model configuration

this is the production model config; a stacking ensemble that beat everything else we tried.

## how we got here

we ran a ton of experiments in `notebooks/2_model_selection.ipynb` comparing different model families and approaches: linear models, random forests, xgboost, lightgbm, and various ensemble methods.

the winner was a stacking ensemble:
- base layer: xgboost + lightgbm (both with early stopping)
- meta layer: ridge regression (alpha=0.001) that learns to combine their predictions

tested on ~4000 trading days (2010-2025), 80/20 train/val split.

## the numbers

validation rmse: **0.00917** (about 0.92% error predicting 5-day volatility)

beats xgboost alone by 0.13% and lightgbm by 0.68%. small gains but consistent.

## what's in this folder

- `model_config.json` - exact hyperparameters for xgboost, lightgbm, and ridge meta-learner
- `benchmark_results.json` - expected performance metrics and monitoring thresholds

## using this config

production training service (`services/train/app.py`) reads these files and builds the ensemble:
1. train xgboost on features
2. train lightgbm on features  
3. stack their predictions
4. train ridge to combine them

logs everything to mlflow for tracking.

---

*exported from notebook on 2025-12-23*
