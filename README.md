# yolo-tsf-comparison

A public showcase of YOLO11-inspired time-series forecasting models benchmarked against PatchTST and DLinear.

## Running

From the `yolo_tsf` folder, an example to run on ETTh1:

`python training_main.py --dataset etth1 --eval-protocol patchtst --repeats 5 --seasonal-kernel 5`

Datasets (`--dataset`): `etth1`, `etth2`, `ettm2`, `ili`, `exchange_rate`, `weather`

## Results

Runs are stored in `yolo_tsf/_OUTPUTS/`. I also report some in `results.md` for a quick comparison
