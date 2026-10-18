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

Evaluate `calibrated.pt` with the same evaluation command. This model uses the 90% group-maximum null-sample scale described below. Insufficient independent units or zero posterior support for nonzero null residuals reject a finite calibrated interval. Pretraining checkpoints can initialize supervised training with `train --init path/to/model.pt`.

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

## Reproduction logic conventions

Held-out calibration scores project reference residuals into each candidate operator null space before posterior marginalization and group-maximum quantile calibration. Observable/range residuals do not inflate the null-sample calibration scale. Exports separate within-operator null variance, between-operator variance and measured within-operator range leakage; remeasurement error remains separate. The finite Gaussian/Fourier operator library and deterministic sampling convention define the supported deblurring setting.

The selected calibration setting preserves the sample-level null-space constraint:
the score is the maximum spatial ratio of operator-weighted absolute projected
null residual to `1.645 * sqrt(null_variance)`. An independent group contributes
its maximum score. The finite-sample 90% quantile, clamped below at one, multiplies
all centered null samples by one global scalar. The model then recomputes null
variance and spatial intervals from the calibrated samples plus the unchanged
operator variance. Observable sample projections, operator-conditioned means,
and between-operator variance stay fixed. This is an explicit reasonable scale
calibration setting; the paper also describes spatial interval expansion. Pixel
interval endpoints do not themselves define an operator-consistent image sample.

Checkpoints record `calibration_mode: null_sample_scale`; earlier calibrated
checkpoints must be recalibrated to avoid interpreting an additive residual as a
scale. A nonzero null residual with zero null variance gives an infinite score,
and inference rejects a finite guarantee in that case. Exported
`operator_samples` and `operator_observable` permit direct projection checks.

## Selected finite restoration setting

`config.json` fixes RGB 64-by-64 images, width 32, Gaussian sigmas 0.7/1.2/2.0,
six deterministic conditional samples, periodic Fourier boundaries and observable
threshold 0.08. Train on paired images degraded by this declared operator family;
the existing demo generates a finite Gaussian-blur problem without external
weights. Use independent source-image groups for fitting, validation, calibration
and test. The group-max null_sample_scale calibration described above is the
selected uncertainty protocol; different blind-SR or spatial-operator families
are outside this chosen experiment definition and are not missing code.

```bash
posteriorrestore demo --output data/example.npz --config config.json --count 96
posteriorrestore prepare --input data/example.npz --output data/prepared
posteriorrestore train --input data/prepared --config config.json --output runs/fourier --variant full --epochs 100 --seed 13
posteriorrestore calibrate --input data/prepared/calibration.npz --model runs/fourier/model.pt --output runs/fourier/calibrated.pt
posteriorrestore evaluate --input data/prepared/test.npz --model runs/fourier/calibrated.pt --output runs/fourier/evaluation
```

At least nine independent calibration groups and nonzero supported null variation
are required for a finite 90% scale. Real paired clean/degraded images are an
optional replacement for the supplied generated image setting.
