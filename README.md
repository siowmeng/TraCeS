# TraCeS

This implementation is based on the official codebase of [omnisafe](https://github.com/PKU-Alignment/omnisafe). It implements the TraCeS method and CT baseline.


## How to install

```bash
conda create --name traces python=3.10
conda activate traces
cd traces/
# Install PyTorch first using the command appropriate for your machine.
# Example for CUDA 12.4; first-time CUDA downloads can be large.
conda install pytorch==2.5.1 pytorch-cuda=12.4 mkl=2023.1.0 intel-openmp=2023.1.0 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```

For CPU-only or a different CUDA version, replace the PyTorch install command with the one recommended for your machine, then run the same `pip install` commands. Human-label and video-rendering workflows additionally require OpenCV: `pip install opencv-python==4.11.0.86`.


## How to run

Train the TraCeS violation estimator before running TraCeS policy training. Provide an HDF5 trajectory dataset from DSRL via `--hdf5_file`. For example:

```bash
python scripts/train_classifier.py \
  -env SafetyHopperVelocity-v1 \
  --hdf5_file /path/to/dsrldata/data/SafetyHopperVelocityGymnasium-v1-250-2240.hdf5 \
  -gruunits 16 \
  -dropout 0.1 \
  -batchsize 128 \
  -targetacc 0.94 \
  -lr 1e-4 \
  -distribution_model \
  -seed 1 \
  -deviceno 0
```

Then set the generated estimator artifact paths in `traces/configs/on-policy/PPOLagTraCeS.yaml` for the corresponding task. The committed YAML files use example paths and should be changed before running policy training:

```yaml
pt_file: path/to/classifier.pt
train_dataset: path/to/traindataset.pt
test_dataset: path/to/testdataset.pt
```

Small pretrained estimator checkpoints may be provided under `artifacts/pretrained_estimators/<task_name>/` using the training-script filename convention, for example `<task_name>_DistributionGRU_4_2_128.pt` for TraCeS and `<task_name>_CostBudgetMLP_128.pt` for CT. The CT implementation still parses the checkpoint filename to identify the estimator class, so keep this convention when adding or replacing CT checkpoints. Large pretraining or online datasets are not included in git; users should generate them locally or point the YAML fields to their own dataset paths. Using a pretrained `pt_file` without the corresponding datasets is possible, but the recommended reproduction path is to train the estimator and save the associated datasets for the task.

Alternatively, set these fields to `null` to skip estimator pretraining and start with a randomly initialized estimator network and empty pretraining dataset. This is useful for debugging or ablations, but pretrained estimator artifacts are the recommended setting for reproducing the paper experiments.

To train a policy using TraCeS:

```bash
python scripts/run_ppo_lag_traces.py [task_name] [start_seed] [end_seed]
```

To train the CT baseline violation estimator, also provide an HDF5 trajectory dataset from DSRL. For example:

```bash
python scripts/train_mlp_classifier.py \
  -env SafetyHalfCheetahVelocity-v1 \
  --hdf5_file /path/to/dsrldata/data/SafetyHalfCheetahVelocityGymnasium-v1-250-2495.hdf5 \
  -dropout 0.0 \
  -batchsize 128 \
  -targetacc 0.96 \
  -lr 1e-3 \
  -seed 9 \
  -deviceno 0
```

Then set the generated estimator artifact paths in `traces/configs/on-policy/PPOLagCT.yaml` for the corresponding task, or set them to `null` if you intentionally want the random-init/no-pretraining estimator mode.

To train a policy using the CT baseline:

```bash
python scripts/run_ppo_lag_ct.py [task_name] [start_seed] [end_seed]
```

