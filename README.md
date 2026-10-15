# PosteriorRestore

Implementation of **PosteriorRestore: Degradation-Marginalized Null-Space Posteriors for Blind Image Restoration**.

A categorical posterior covers a finite library of Fourier blur operators. Each sampled operator has an exact thresholded spectral pseudoinverse and complementary observable/null projectors. Learned image-prior perturbations are projected into the null component, and uncertainty separates within-operator null variation from between-operator variation. The operator library, spectral threshold, and Monte Carlo count are explicit implementation choices.

## Install

Use Python 3.10 or later. Create and activate a virtual environment, then install:

```bash
python -m pip install -e .
```

## Prepare data

`x[N,3,H,W]` and `y[N,3,H,W]` are degraded and clean paired images. Images use periodic Fourier boundaries. `sigma` in the configuration specifies candidate Gaussian blur scales; replace these with characterized operators for a different image formation model.

Arrays use numeric NPZ format, loaded without pickle. `group[N]` identifies independent subjects, devices, scenes, lots, or source images. Related samples must share a group. Optional `split[N]` codes 0/1/2/3 explicitly assign training/validation/calibration/test (three-role input using 0/1/2 remains supported); otherwise ordered groups are allocated 60/15/10/15 to training/validation/calibration/test. Test data never select model parameters or checkpoints.

```bash
posteriorrestore demo --output data/example.npz --count 96 --config config.json
posteriorrestore prepare --input data/example.npz --output data/prepared
```

The demo defines a small synthetic problem with the documented tensor interface. Use application data in that schema for the paper's evaluation settings.

Paired image folders with matching filenames can be packed with:

```bash
posteriorrestore pack --images data/images --targets data/targets --mask --output data/images.npz --size 64
```

Omit `--mask` for RGB reconstruction targets. Add additional required arrays according to the schema above before splitting.

## Train and execute

```bash
posteriorrestore train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16
posteriorrestore evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
posteriorrestore plot --input runs --output runs/figures
```

Training uses AdamW and selects the minimum validation objective. `model.pt` stores parameters and configuration; `history.json` records training/validation objectives. Evaluation exports predictions and mechanism tensors to `predictions.npz` and computed scores to `metrics.json`. Plotting exports comparison charts from those actual evaluation files.

`--device cpu` is the default; `--device cuda` selects CUDA. `--lr`, `--epochs`, `--batch-size`, and `--seed` control optimization. Method parameters are in `config.json`; `--set key=value` overrides one value using JSON syntax. Use identical splits and seeds across comparisons.

## Calibration

For interval or classification confidence calibration, use independent groups excluded from fitting and checkpoint selection:

```bash
posteriorrestore calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
```

Evaluate `calibrated.pt` with the same evaluation command. Classification uses temperature scaling; numeric intervals use a 90% group-maximum split-calibration expansion. Insufficient independent units retain infinite intervals. Pretraining checkpoints can initialize supervised training with `train --init path/to/model.pt`.

## Baselines and ablations

Available variants: `full`, `baseline`, `point_operator`, `no_projection`. `baseline` removes the paper-specific mechanism while retaining the input and prediction task. Other variants disable their named component.

```bash
posteriorrestore train --input data/prepared --output runs/baseline --config config.json --variant baseline
posteriorrestore evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Metrics and implementation conventions

Classification reports accuracy, balanced accuracy, macro-F1, NLL, confidence ECE, and worst-group F1; binary tasks add AUROC and AUPRC. Unknown labels use -1 and enable unknown AUROC and OSCR. Segmentation reports Dice, IoU, precision, recall, MAE, Brier/ECE, two-pixel boundary F1, symmetric contour distance, and HD95. Reconstruction reports RMSE, MAE, signed bias, unit-range PSNR, and Gaussian-window SSIM. Each method adds its mechanism metrics where applicable.

Parameter counts and forward timing describe the executing process. Use batch size one and a declared device configuration for device latency comparisons. Physical energy and scores from external task evaluators require measured inputs; they are not inferred from parameter counts. All prediction arrays remain available for task-specific evaluation.

The implementation uses the explicit core operations described above with configurable compact networks. Numeric conventions and application-specific information are stated with the method and data schema. The parameter file contains the current defaults.

## Perceptual and saliency endpoints

```bash
python -m pip install -e ".[image-metrics]"
python -m posteriorrestore.evaluate_images --predictions runs/full/evaluation/predictions.npz --reference data/prepared/test.npz --metric lpips --output runs/additional_metrics.json
```

LPIPS uses the [published perceptual metric implementation](https://github.com/richzhang/PerceptualSimilarity) with RGB values mapped to [-1,1]. Saliency endpoints use [PySODMetrics](https://github.com/lartpang/PySODMetrics) for adaptive/maximum F-measure, E-measure, S-measure, and weighted F-measure; their threshold conventions remain those of the metric package.
