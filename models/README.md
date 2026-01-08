# Models Directory Documentation

This directory contains trained models, baselines, and historical tracking files for the volatility forecasting system.

## 📁 File Overview

### Trained Model Files (`.pkl`)

| File | Description | Updated | Purpose |
|------|-------------|---------|---------|
| `model_90pct.pkl` | **Model A** - Conservative model trained on 90% of data | Every training run | Production model with true out-of-sample validation |
| `model_100pct.pkl` | **Model B** - Aggressive model trained on all data except last 30 samples | Every training run | Alternative model using maximum available data |
| `latest_ensemble.pkl` | Symlink/copy of `model_90pct.pkl` | Every training run | Backward compatibility for services expecting this name |

### Baseline Files (`.json`)

| File | Description | Updated | Purpose |
|------|-------------|---------|---------|
| `validation_baseline.json` | Model A validation metrics and date ranges | Every training run (overwritten) | Used by monitor service for drift detection |
| `dual_model_baseline.json` | Both Model A and Model B training info and metrics | Every training run (overwritten) | Used by monitor service for model comparison |

### Historical Tracking Files (`.jsonl`)

| File | Description | Updated | Purpose |
|------|-------------|---------|---------|
| `training_history.jsonl` | Complete history of all training runs | Every training run (appended) | Track model performance trends over time |
| `monitoring_history.jsonl` | Complete history of all monitoring runs | Every monitoring run (appended) | Track drift and model comparison over time |

---

## 📊 Detailed File Specifications

### 1. Model Files (`model_90pct.pkl`, `model_100pct.pkl`)

**Format**: Python pickle (serialized dictionary)

**Contents**:
```python
{
    'xgb_model': <XGBoost model>,
    'lgbm_model': <LightGBM model>,
    'meta_model': <Ridge meta-learner>,
    'feature_cols': ['spy_ret_1d', 'spy_ret_5d', ...]  # List of 20 features
}
```

**Usage**:
```python
import pickle

# Load model
with open('models/model_90pct.pkl', 'rb') as f:
    model_artifact = pickle.load(f)

# Make predictions
X_meta = np.column_stack([
    model_artifact['xgb_model'].predict(X),
    model_artifact['lgbm_model'].predict(X)
])
predictions = model_artifact['meta_model'].predict(X_meta)
```

**Size**: ~1.5 MB each

**Notes**:
- Model A (90%): Trained on 90% of data, validated on held-out 10%
- Model B (All-30): Trained on all data except last 30 samples for immediate testing
- Both are ensemble models: XGBoost + LightGBM + Ridge meta-learner

---

### 2. `validation_baseline.json`

**Format**: JSON (single object)

**Purpose**: Stores Model A's validation performance for drift detection

**Schema**:
```json
{
  "validation_rmse": 0.011402,
  "validation_mae": 0.006778,
  "validation_r2": 0.434352,
  "timestamp": "2026-01-08T22:09:13.586357",
  "train_val_split": 0.9,
  "date_range": {
    "train_start": "2010-03-31",
    "train_end": "2024-05-31",
    "val_start": "2024-06-03",
    "val_end": "2025-12-31"
  }
}
```

**Updated**: Every training run (file overwritten)

**Used By**: Monitor service to detect performance drift

**Key Fields**:
- `validation_rmse`: Baseline RMSE on validation set (10% holdout)
- `date_range`: Critical for knowing what data was NOT used in training
- `timestamp`: When this baseline was established

---

### 3. `dual_model_baseline.json`

**Format**: JSON (single object)

**Purpose**: Stores both Model A and Model B training information for comparison

**Schema**:
```json
{
  "timestamp": "2026-01-08T22:09:13.586357",
  "model_a": {
    "training_samples": 3566,
    "training_date_start": "2010-03-31",
    "training_date_end": "2024-05-31",
    "validation_rmse": 0.011402,
    "validation_mae": 0.006778,
    "validation_r2": 0.434352,
    "validation_samples": 397
  },
  "model_b": {
    "training_samples": 3933,
    "training_date_start": "2010-03-31",
    "training_date_end": "2025-11-17",
    "test_samples": 30,
    "test_date_start": "2025-11-18",
    "test_date_end": "2025-12-31",
    "test_rmse": 0.005797,
    "test_mae": 0.004760,
    "test_r2": -0.122300,
    "note": "Model B held out last 30 samples for testing"
  }
}
```

**Updated**: Every training run (file overwritten)

**Used By**: Monitor service for model comparison and date filtering

**Critical Fields**:
- `training_date_end`: Last date used in training (monitor uses this for filtering)
- `test_date_start/end`: Date range for Model B's held-out test set
- Performance metrics: Used to compare training iterations over time

---

### 4. `training_history.jsonl`

**Format**: JSONL (JSON Lines - one JSON object per line)

**Purpose**: Append-only history of all training runs for trend analysis

**Schema** (one line per training run):
```json
{
  "timestamp": "2026-01-08T22:09:13.586357",
  "training_date": "2026-01-08",
  "model_a": {
    "training_samples": 3566,
    "training_date_start": "2010-03-31",
    "training_date_end": "2024-05-31",
    "validation_rmse": 0.011402,
    "validation_mae": 0.006778,
    "validation_r2": 0.434352,
    "validation_samples": 397
  },
  "model_b": {
    "training_samples": 3933,
    "training_date_start": "2010-03-31",
    "training_date_end": "2025-11-17",
    "test_samples": 30,
    "test_date_start": "2025-11-18",
    "test_date_end": "2025-12-31",
    "test_rmse": 0.005797,
    "test_mae": 0.004760,
    "test_r2": -0.122300,
    "note": "Model B held out last 30 samples for testing"
  }
}
```

**Updated**: Every training run (new line appended)

**Growth**: Grows indefinitely (one line per training run)

**Analysis Examples**:

```python
import json
import pandas as pd

# Load all training history
with open('models/training_history.jsonl') as f:
    history = [json.loads(line) for line in f]

# Convert to DataFrame for analysis
df = pd.DataFrame(history)

# Track Model A validation RMSE over time
model_a_rmse = [r['model_a']['validation_rmse'] for r in history]
print(f"Model A RMSE trend: {model_a_rmse}")

# Find best performing training run
best_run = min(history, key=lambda x: x['model_a']['validation_rmse'])
print(f"Best Model A RMSE: {best_run['model_a']['validation_rmse']}")
```

**Use Cases**:
- Track if retraining improves performance
- Identify performance degradation over time
- Compare training runs from different dates
- Audit trail of all model versions

---

### 5. `monitoring_history.jsonl`

**Format**: JSONL (JSON Lines - one JSON object per line)

**Purpose**: Append-only history of all monitoring runs for drift and comparison tracking

**Schema** (one line per monitoring run):
```json
{
  "timestamp": "2026-01-08T22:03:39.666322",
  "monitoring_date": "2026-01-08",
  "model_a": {
    "training_samples": 3566,
    "training_date_start": "2010-03-31",
    "training_date_end": "2024-05-31"
  },
  "model_b": {
    "training_samples": 3933,
    "training_date_start": "2010-03-31",
    "training_date_end": "2025-11-17"
  },
  "test_window": {
    "n_samples": 30,
    "date_start": "2025-11-18",
    "date_end": "2025-12-31"
  },
  "drift": {
    "model_a_drift_pct": 0.0
  },
  "comparison": {
    "model_a_rmse": 0.005765,
    "model_b_rmse": 0.005768,
    "winner": "Model A",
    "model_b_win_rate": 0.0
  },
  "recommendation": "USE MODEL A (90%)",
  "windows": [
    {
      "name": "All new data (30 samples)",
      "n_samples": 30,
      "model_a_rmse": 0.005765,
      "model_b_rmse": 0.005768,
      "winner": "Model A"
    },
    ...
  ]
}
```

**Updated**: Every monitoring run (new line appended)

**Growth**: Grows indefinitely (one line per monitoring run)

**Analysis Examples**:

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load all monitoring history
with open('models/monitoring_history.jsonl') as f:
    history = [json.loads(line) for line in f]

# Track drift over time
dates = [r['monitoring_date'] for r in history]
drift = [r['drift']['model_a_drift_pct'] for r in history]

plt.plot(dates, drift)
plt.title('Model A Drift Over Time')
plt.ylabel('Drift %')
plt.show()

# Model comparison win rates over time
model_b_wins = [r['comparison']['model_b_win_rate'] for r in history]
print(f"Model B average win rate: {sum(model_b_wins)/len(model_b_wins):.1f}%")
```

**Use Cases**:
- Detect when drift exceeds thresholds
- Track which model performs better over time
- Alert when retraining is needed
- Analyze seasonal patterns in model performance

---

## 🔄 File Lifecycle

### Training Workflow

```mermaid
Training Run
    ├─→ Overwrites: model_90pct.pkl
    ├─→ Overwrites: model_100pct.pkl
    ├─→ Overwrites: latest_ensemble.pkl
    ├─→ Overwrites: validation_baseline.json
    ├─→ Overwrites: dual_model_baseline.json
    └─→ Appends to: training_history.jsonl  ← HISTORICAL RECORD
```

### Monitoring Workflow

```mermaid
Monitoring Run
    ├─→ Reads: validation_baseline.json
    ├─→ Reads: dual_model_baseline.json
    ├─→ Reads: model_90pct.pkl
    ├─→ Reads: model_100pct.pkl
    └─→ Appends to: monitoring_history.jsonl  ← HISTORICAL RECORD
```

---

## 🎯 Common Questions

### Q: Which file should I use for production predictions?

**A**: Use `model_90pct.pkl` (Model A) by default. It has true out-of-sample validation. Only switch to `model_100pct.pkl` (Model B) if monitoring shows it consistently outperforms Model A (≥75% win rate AND drift < 10%).

### Q: How do I know if my model needs retraining?

**A**: Check `monitoring_history.jsonl`:
- If `drift.model_a_drift_pct` > 10% → Retrain immediately
- If `drift.model_a_drift_pct` 5-10% → Monitor closely, plan retraining
- If performance degrades consistently over 3+ monitoring runs → Retrain

### Q: Can I delete old history files?

**A**: No! These are append-only audit trails. If you need to archive:
```bash
# Archive history (keep last 1000 lines)
tail -1000 training_history.jsonl > training_history_recent.jsonl
mv training_history.jsonl training_history_archive_$(date +%Y%m%d).jsonl
mv training_history_recent.jsonl training_history.jsonl
```

### Q: How do I compare training runs from different weeks?

**A**: Parse `training_history.jsonl`:
```python
import json

with open('models/training_history.jsonl') as f:
    runs = [json.loads(line) for line in f]

# Compare first and last run
print(f"First run RMSE: {runs[0]['model_a']['validation_rmse']:.6f}")
print(f"Latest run RMSE: {runs[-1]['model_a']['validation_rmse']:.6f}")
print(f"Change: {((runs[-1]['model_a']['validation_rmse'] / runs[0]['model_a']['validation_rmse']) - 1) * 100:.2f}%")
```

### Q: What's the difference between Model A and Model B?

| Aspect | Model A (90%) | Model B (All-30) |
|--------|--------------|------------------|
| Training data | 90% of dataset (3,566 samples) | All data except last 30 (3,933 samples) |
| Validation | True out-of-sample (397 samples never seen) | Last 30 samples held out for testing |
| Use case | Conservative, reliable baseline | Aggressive, uses maximum data |
| When to use | Default production model | When empirically proven better |

---

## 🛡️ Data Safety

### Backup Strategy

**Critical files** (models are expensive to retrain):
- `model_90pct.pkl`
- `model_100pct.pkl`
- `training_history.jsonl`
- `monitoring_history.jsonl`

**Recommendation**:
```bash
# Backup before retraining
tar -czf models_backup_$(date +%Y%m%d_%H%M%S).tar.gz models/
```

### Version Control

**Do NOT commit**:
- `.pkl` files (too large, binary)
- History `.jsonl` files if very large (>10MB)

**DO commit**:
- This README.md
- Small baseline `.json` files (if < 1MB)

**Add to `.gitignore`**:
```
models/*.pkl
models/*_history.jsonl
```

---

## 📈 Monitoring Best Practices

1. **Daily monitoring**: Run `docker-compose up monitor` daily to track drift
2. **Weekly training**: Retrain models weekly to keep them fresh
3. **Review trends**: Check `monitoring_history.jsonl` weekly for patterns
4. **Alert thresholds**:
   - Drift > 10% → Immediate retraining
   - Drift 5-10% → Plan retraining within 3 days
   - Model B win rate > 75% for 3+ consecutive runs → Consider switching

---

## 🔧 Maintenance

### Cleanup Old Models

If you're retraining frequently and want to save space:

```bash
# Archive old models before retraining
mkdir -p models/archive/$(date +%Y%m)
cp models/model_*.pkl models/archive/$(date +%Y%m)/
```

### Analyze Performance Trends

```python
#!/usr/bin/env python3
"""Analyze training and monitoring trends"""

import json
import pandas as pd

# Load training history
with open('models/training_history.jsonl') as f:
    training = pd.DataFrame([json.loads(line) for line in f])

# Load monitoring history  
with open('models/monitoring_history.jsonl') as f:
    monitoring = pd.DataFrame([json.loads(line) for line in f])

# Training trends
print("=== TRAINING TRENDS ===")
print(f"Total training runs: {len(training)}")
if len(training) > 1:
    first_rmse = training.iloc[0]['model_a']['validation_rmse']
    last_rmse = training.iloc[-1]['model_a']['validation_rmse']
    change = ((last_rmse / first_rmse) - 1) * 100
    print(f"RMSE change: {change:+.2f}%")

# Monitoring trends
print("\n=== MONITORING TRENDS ===")
print(f"Total monitoring runs: {len(monitoring)}")
if len(monitoring) > 0:
    avg_drift = monitoring['drift'].apply(lambda x: x['model_a_drift_pct']).mean()
    print(f"Average drift: {avg_drift:.2f}%")
```

---

## 📚 Related Documentation

- **Training Service**: `services/train/README.md` - How models are trained
- **Monitoring Service**: `services/monitor/README.md` - How monitoring works
- **Main README**: `README.md` - Overall system architecture

---

**Last Updated**: 2026-01-08  
**Maintained By**: Volatility Forecasting Team
