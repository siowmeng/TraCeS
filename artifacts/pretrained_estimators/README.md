# Pretrained Estimator Checkpoints

Optional pretrained estimator checkpoints can be placed here for release convenience. Use one subdirectory per task, preserving the training-script filename convention, for example:

```text
artifacts/pretrained_estimators/SafetyHalfCheetahVelocity-v1/SafetyHalfCheetahVelocity-v1_DistributionGRU_4_2_128.pt
artifacts/pretrained_estimators/SafetyHalfCheetahVelocity-v1/SafetyHalfCheetahVelocity-v1_CostBudgetMLP_128.pt
```

Only small model checkpoint files should be committed here. Large pretraining or online datasets such as `*_traindataset.pt` and `*_testdataset.pt` are not provided; users should generate them locally (when training the estimator) and point the YAML fields to their own dataset paths.
