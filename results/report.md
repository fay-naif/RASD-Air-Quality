# RASD: Evaluation Report

## Data and target

Source: UCI Beijing Multi-Site Air Quality, Aotizhongxin station, 2013-2017.
The app uses 35,064 original observations and the CSV in `data/air_quality.csv`.

The forecasting target is PM2.5 one hour ahead. The classification target is the following-hour PM2.5 class:

| Class | PM2.5 range |
|---|---:|
| Low | <= 12 µg/m³ |
| Elevated | >12 and <=35.4 µg/m³ |
| High | >35.4 and <=55.4 µg/m³ |
| Very high | >55.4 µg/m³ |

These classes are research categories, not official health alerts.

## Chronological evaluation

The prepared observations use a chronological 70/15/15 split:

| Split | Rows |
|---|---:|
| Training | 23,317 |
| Validation | 4,997 |
| Held-out test | 4,997 |

Lag and rolling features use only previous observations. The target is excluded from predictors. Learned preprocessing is not fitted across split boundaries.

## Classification results

Selected model: class-weighted Logistic Regression with median imputation and scaling.

| Metric | Held-out test |
|---|---:|
| Accuracy | 79.69% |
| Balanced accuracy | 77.96% |
| Macro F1 | 74.08% |

Run `python train_classification.py` to regenerate the model and these values in the active environment.

## Forecast results

The numerical forecast model was rebuilt in the active environment.

| Metric | Held-out test |
|---|---:|
| MAE | 10.723 µg/m³ |
| RMSE | 20.576 µg/m³ |
| R² | 0.9527 |

These values are the same values stored in `results/metrics.json`.

## Scope and limitations

Results describe one historical monitoring station and a chronological holdout. They do not establish real-time or multi-city performance. The interface is retrospective and must not be used as a medical warning system.
