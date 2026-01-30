# yolo-tsf-comparison

In the `yolo_tsf` folder, run with:

`python training_etth1.py --no-skip`

- (add `--no-skip` since the skip connection may not work correctly with RevIN)

You can adjust the hyperparameters or see other CLI flags at the top of `training_etth1.py`.

---

Current output (window=512, horizon=24, epochs=100, learning rate=3e-4, batch size=64):

```powershell
======================================================================
TRAINING ALL MODELS (use_skip=False, use_revin=True, patch_len=16, patch_stride=8)
======================================================================

Training: YOLO11 Forecast
----------------------------------------
  Best Val Loss: 2.5701
  Test MSE: 3.7607 | Test MAE: 1.4773

Training: v1 - Backbone Only
----------------------------------------
  Best Val Loss: 2.6871
  Test MSE: 3.8177 | Test MAE: 1.4972

Training: v2 - Backbone + Neck
----------------------------------------
  Best Val Loss: 2.7193
  Test MSE: 3.7648 | Test MAE: 1.4866

Training: v3 - Full (Multiscale)
----------------------------------------
  Best Val Loss: 2.5076
  Test MSE: 3.7682 | Test MAE: 1.4429

======================================================================
RESULTS SUMMARY
======================================================================
Model                         Val Loss     Test MSE     Test MAE
----------------------------------------------------------------------
YOLO11 Forecast                 2.5701       3.7607       1.4773
v1 - Backbone Only              2.6871       3.8177       1.4972
v2 - Backbone + Neck            2.7193       3.7648       1.4866
v3 - Full (Multiscale)          2.5076       3.7682       1.4429
----------------------------------------------------------------------
Best (by Test MSE): YOLO11 Forecast
```
