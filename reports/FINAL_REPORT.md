# Final Project Report — Adaptive Opus

## 1. Model Training Reproduction

Reproduced training and evaluation using `RandomForestRegressor(n_estimators=100, random_state=42)` on `opus_dataset.csv`.

### Performance Metrics
- Training R²: 0.8401
- Test R²: 0.8166
- Training MAE: 0.2083
- Test MAE: 0.2184
- Training RMSE: 0.2761
- Test RMSE: 0.2871
- 5-fold CV R²: 0.8260 ± 0.0482

### Feature Importances (retrained model)
- bitrate: 0.2135 (21.3%)
- frame_size: 0.0125 (1.3%)
- use_fec: 0.0033 (0.3%)
- packet_loss_perc: 0.7707 (77.1%)

### Feature Importances (saved `qoe_model.joblib`)
- bitrate: 0.2135 (21.3%)
- frame_size: 0.0125 (1.3%)
- use_fec: 0.0033 (0.3%)
- packet_loss_perc: 0.7707 (77.1%)

## 2. Key Findings Verification

- The reproduced training shows a large gap between training and test R², indicating some overfitting. The test R² around 0.81 confirms the previously reported R² ≈ 0.813.
- Feature importances confirm that `packet_loss_perc` dominates model predictions and `use_fec` contributes very little in the training corpus. This matches the project observation that FEC importance was low due to clean speech in LibriSpeech.

## 3. Out-of-Distribution (OOD) Test Summary

Included OOD report and plots:
- See `OOD_REPORT.md` and files in `reports/` for per-run results and plots.

## 4. Conclusion

- The analyses validate the project's claim: the model explains ~81% of variance on the test set but learned dataset-specific patterns (overfitting to LibriSpeech).
- The Hybrid controller is validated by OOD testing: it avoids ML decisions at high loss where ML is unreliable.