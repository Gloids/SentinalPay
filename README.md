# Two-Stage Fraud Detection: Real-Time Anomaly Scoring + Batch Ensemble Classification

A production-oriented fraud detection system that resolves the fundamental tension between **real-time speed** and **classification accuracy** by splitting the problem across two models operating at different points in the transaction lifecycle.

---

## The Problem

Every fraud detection system faces two competing constraints:

- **Speed** — Payment processors need a decision in under 100ms. Models must run in fractions of a millisecond.
- **Accuracy** — Fraud is rare (3.5% of transactions). High recall requires large, slow ensemble models that can't fit inside the real-time window.

Most systems compromise one for the other. This architecture doesn't.

---

## The Solution: Two Stages, Two Timings
```
Every Transaction
       │
       ▼
┌─────────────────────────────┐
│  STAGE 1: Isolation Forest  │  ← T = 0.046 ms  (real-time, during authorisation)
│  Unsupervised anomaly gate  │    Runs on 100% of transactions
└────────────┬────────────────┘
             │
    ┌─────────┴──────────┐
    │                    │
    ▼                    ▼
 BLOCK               APPROVE
(suspicious)        (looks normal)
    │                    │
    └─────────┬──────────┘
              │  Transaction settles  (T + minutes)
              ▼
┌─────────────────────────────┐
│  STAGE 2: Random Forest     │  ← T = 0.857 ms  (batch, after settlement)
│  Supervised classifier      │    Runs on 100% of transactions
└────────────┬────────────────┘
             │
    ┌─────────┴──────────┐
    │                    │
    ▼                    ▼
 FLAG FOR            CLEARED
 INVESTIGATION
    │
    └──────────────────────────► Feedback loop:
                                  Stage 2 labels missed frauds
                                  → retrains Stage 1
                                  → recall improves each cycle
```

---

## Key Results

| Stage | Speed | Recall | Notes |
|---|---|---|---|
| Stage 1 (Isolation Forest) | **0.046 ms/txn** | 65.7% | Real-time, no labels needed |
| Stage 2 (Random Forest) @ thr 0.010 | 0.857 ms/txn | **95.7%** | Batch, post-settlement |
| **Full System** | Both stages | **96.3%** | Combined coverage |

**Speed gap: Stage 2 is 19x slower per transaction** — which is exactly the point. Stage 1 fits inside the payment authorisation window. Stage 2 doesn't need to.

### Fraud Flow Breakdown (4,133 total fraud cases in test set)

| Outcome | Count | % of Fraud |
|---|---|---|
| Stage 1 blocks in real-time | 2,716 | 65.7% |
| Stage 1 misses → Stage 2 recovers | 1,263 | 30.6% |
| Both miss (feedback target) | 177 | 4.3% |
| **Total caught** | **3,979** | **96.3%** |

### Scalability

| | Stage 1 | Stage 2 |
|---|---|---|
| Inference time | 0.046 ms | 0.857 ms |
| Single instance | 21,596 txn/sec | 1,167 txn/sec |
| Realistic cluster | 100 instances = **2.16M txn/sec** | 3 instances = **3,500 txn/sec** |
| Deployment mode | Always-on real-time | Batch after settlement |

Stage 1 is cheap and horizontally scalable across commodity servers. Stage 2 is heavy but only needs a few instances running in batch mode.

---

## Architecture Details

### Stage 1 — Isolation Forest (Real-Time Gate)

- **Algorithm**: Isolation Forest (unsupervised — no fraud labels required)
- **Config**: 50 trees, 30% sample per tree
- **Why small**: Fewer trees = faster inference. 50 trees give 4x speedup over 200 with modest accuracy reduction
- **Threshold**: Flags top 32% most anomalous transactions
- **Training time**: 1.72 seconds
- **Output**: Anomaly score (continuous) + suspicious flag (binary) → passed to Stage 2 as features

### Stage 2 — Random Forest (Batch Classifier)

- **Algorithm**: Random Forest (supervised — uses fraud labels)
- **Config**: 500 trees, unlimited depth, class weight 1:100
- **Why heavy**: More trees + extreme class weighting pushes recall to 95%+
- **Three operating points**:

  | Threshold | Recall | Precision | F1 | Use case |
  |---|---|---|---|---|
  | 0.50 (default) | 40.2% | 94.8% | 56.4% | When false alarms are expensive |
  | 0.193 (max F1) | 64.2% | 74.6% | 69.0% | General investigative workflow |
  | 0.010 (95% recall) | 95.7% | 7.6% | 14.0% | Maximum fraud coverage |

- **ROC-AUC**: 0.9447 — strong ranking quality across all thresholds
- **Training time**: 328.8 seconds
- **Input features**: 405 (403 transaction features + Stage 1 anomaly score + Stage 1 flag)

### Feedback Loop (Architectural, Not Yet Implemented in Code)

Stage 2 identifies frauds that Stage 1 missed. These 1,263 cases per evaluation cycle become labelled positive training examples for the next Stage 1 retraining. Over successive cycles, Stage 1 recall improves as it learns patterns that are currently invisible to unsupervised detection. The 177 cases both stages miss are the primary target for progressive reduction.

---

## Dataset

- **590,540** e-commerce transactions
- **3.5% fraud rate** (20,663 fraudulent, 569,877 legitimate)
- **394 raw features**: transaction metadata, card attributes, email domains, count features (C1-C14), time-delta features (D1-D15), match flags (M1-M9), and 339 proprietary engineered features (V1-V339)
- **80/20 stratified split**: 472,432 train / 118,108 test

### Feature Engineering (9 new features added)

| Feature | Description |
|---|---|
| `Amount_log` | log(1 + amount) — normalises right-skewed distribution |
| `Amount_log2` | log(1 + amount²) — amplifies high-value differences |
| `Amount_bin` | Quartile bucket (0-3) |
| `Amount_round` | 1 if amount has no cents — fraud behavioural signal |
| `Time_hour` | Hour of day (0-23) |
| `Time_day` | Day index from reference date |
| `Time_weekend` | 1 if weekend |
| `Hour_sin` | sin(2π × hour / 24) — cyclic encoding |
| `Hour_cos` | cos(2π × hour / 24) — cyclic encoding |

---

## Project Structure
```
├── sentinelpay_train.py       # Train both Stage 1 and Stage 2 models
├── sentinelpay_viz.py         # Generate all visualizations
├── models_sentinelpay/
│   ├── pre_isolation_forest.joblib
│   ├── post_random_forest.joblib
│   ├── scaler.joblib
│   ├── label_encoders.joblib
│   └── config.json            # All metrics, thresholds, sweep results
├── visualizations_sentinelpay/
│   ├── 01_architecture.png
│   ├── 02_latency_vs_accuracy.png
│   ├── 03_scalability.png
│   ├── 04_timeline.png
│   ├── 05_stage1_analysis.png
│   ├── 06_stage2_threshold.png
│   ├── 07_stage2_confusion.png
│   ├── 08_stage2_curves.png
│   ├── 09_fraud_flow.png
│   └── ...
└── logs_sentinelpay/
    └── training_YYYYMMDD_HHMMSS.log
```

---

## Setup and Usage

### 1. Create environment
```bash
conda create -n fraud_ieee python=3.10 -y
conda activate fraud_ieee
conda install -c conda-forge numpy pandas scikit-learn joblib matplotlib seaborn -y
```

### 2. Configure paths

Edit the `TRAIN_PATH` variable at the top of `sentinelpay_train.py`:
```python
TRAIN_PATH = "path/to/your/train_transaction.csv"
```

### 3. Train
```bash
python sentinelpay_train.py
```

Training takes approximately 6-7 minutes on a standard machine (mostly Stage 2 — 328 seconds for 500 trees).

### 4. Generate visualizations
```bash
python sentinelpay_viz.py
```

All 13 charts saved to `visualizations_sentinelpay/`.

---

## Why This Architecture

### The latency argument

A single model cannot occupy the top-left corner of the latency-accuracy trade-off space simultaneously. Stage 1 sits at (0.046 ms, 65.7% recall) — fast enough for real-time, good enough to block the majority of fraud. Stage 2 sits at (0.857 ms, 95.7% recall) — too slow for real-time, accurate enough for post-hoc analysis. Neither is useful alone; together they cover 96.3% of fraud.

### The scalability argument

Stage 1's lightweight design allows 100 cheap commodity instances to collectively process 2.16 million transactions per second — sufficient for global payment network volumes. Stage 2 runs on 3 heavier instances in batch, processing the same transaction volume overnight. The total infrastructure cost is asymmetric but appropriate: the expensive computation (Stage 2) runs offline where time is not a constraint.

### The feedback argument

Without a feedback mechanism, the 177 frauds both stages miss would continue to evade detection indefinitely. The feedback loop converts Stage 2's post-hoc findings into Stage 1 training signal, progressively closing the gap. Each deployment cycle should reduce the 4.3% residual miss rate.

---

## Visualizations

| Chart | What it shows |
|---|---|
| `02_latency_vs_accuracy` | **Key chart** — log-scale scatter showing 19x speed gap |
| `03_scalability` | Inference time, throughput, cluster comparison |
| `04_timeline` | When each stage acts in a transaction's lifecycle |
| `05_stage1_analysis` | Score distribution + threshold trade-off |
| `06_stage2_threshold` | F1, Precision, Recall vs threshold for Stage 2 |
| `07_stage2_confusion` | Confusion matrices at 3 operating points |
| `08_stage2_curves` | ROC and Precision-Recall curves |
| `09_fraud_flow` | Stage 1 vs Stage 2 fraud contribution breakdown |
| `12_summary_dashboard` | All key numbers in one view |

---

## Academic Reference

> A Two-Stage Real-Time and Batch Framework for Transactional Fraud Detection Using Anomaly Scoring and Ensemble Classification. Major Project, Department of Computer Science and Engineering, 2026.

---

## License

This project is submitted as academic work. Dataset usage subject to original dataset terms and conditions.
