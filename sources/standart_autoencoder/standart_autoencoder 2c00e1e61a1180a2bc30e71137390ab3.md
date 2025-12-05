# standart_autoencoder

`{
  "train_root": "/content/symmetry_dataset",
  "test_root": "/content/test",
  "output_dir": "/content/artifacts",
  "image_size": [
    256,
    256
  ],
  "num_channels": 3,
  "normalize": true,
  "batch_size": 32,
  "num_workers": 4,
  "pin_memory": true,
  "val_split": 0.1,
  "epochs": 30,
  "learning_rate": 0.001,
  "weight_decay": 1e-05,
  "max_grad_norm": 5.0,
  "patience": 5,
  "max_train_samples": 85000,
  "max_test_samples": null,
  "latent_channels": 64,
  "base_channels": 32,
  "seed": 42,
  "in_channels": 3,
  "normalize_mean": [
    0.5,
    0.5,
    0.5
  ],
  "normalize_std": [
    0.5,
    0.5,
    0.5
  ]
}`

Training samples: 76500
Validation samples: 8500
Test samples: 355
Number of classes: 17
Class names: ['cm', 'cmm', 'p1', 'p2', 'p3', 'p31m', 'p3m1', 'p4', 'p4g', 'p4m', 'p6', 'p6m', 'pg', 'pgg', 'pm', 'pmg', 'pmm']

![image.png](standart_autoencoder/image.png)

`Model parameter count: 464,739
Model size: 0.46 M parameters

Model Architecture:
ConvAutoencoder(
  (encoder): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace=True)
    (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): LeakyReLU(negative_slope=0.2, inplace=True)
    (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): LeakyReLU(negative_slope=0.2, inplace=True)
    (9): Conv2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): LeakyReLU(negative_slope=0.2, inplace=True)
  )
  (decoder): Sequential(
    (0): ConvTranspose2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace=True)
    (9): ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (10): Tanh()
  )
)`

Epoch 21: train_loss=0.009903, val_loss=0.009579
Early stopping triggered.
CPU times: user 1h 6min 40s, sys: 5min 41s, total: 1h 12min 21s
Wall time: 1h 7min 50s

Best model loaded (epoch=16, val_loss=0.009285)

![image.png](standart_autoencoder/image%201.png)

============================================================
TRAINING SUMMARY
============================================================
Total Epochs: 21
Final Training Loss: 0.009903
Final Validation Loss: 0.009579
Best Validation Loss: 0.009285
============================================================

`Test MSE Loss: 0.060758

============================================================
CLASS-WISE RECONSTRUCTION ERROR (mean ± std)
============================================================
  p1                  : 0.030025 ± 0.027309
  pm                  : 0.036713 ± 0.030132
  cm                  : 0.043554 ± 0.033758
  p3m1                : 0.046524 ± 0.038429
  pmg                 : 0.049887 ± 0.038073
  p3                  : 0.052804 ± 0.043831
  p31m                : 0.053864 ± 0.036063
  p2                  : 0.058557 ± 0.058189
  p4m                 : 0.058996 ± 0.035134
  pg                  : 0.061262 ± 0.060334
  cmm                 : 0.062279 ± 0.075607
  p4g                 : 0.064847 ± 0.051951
  pmm                 : 0.068911 ± 0.051598
  pgg                 : 0.074798 ± 0.081493
  p6m                 : 0.087812 ± 0.075404
  p4                  : 0.107977 ± 0.126374
  p6                  : 0.110968 ± 0.096030
============================================================`

![image.png](standart_autoencoder/image%202.png)

Latent space shape: (355, 16384)
Number of samples: 355
Unique classes: 17
Running t-SNE on 355 samples...

![image.png](standart_autoencoder/image%203.png)

Running PCA on 355 samples…

![image.png](standart_autoencoder/image%204.png)

PCA Analysis:
  - First 2 components explain 14.1% of variance
  - 272 components needed to explain 95% of variance

![image.png](standart_autoencoder/image%205.png)

# ============================================================
ERROR DISTRIBUTION STATISTICS

# Total samples: 355
Mean error: 0.060758
Std error: 0.065446
Min error: 0.000499
Max error: 0.513804
Median error: 0.038702

![image.png](standart_autoencoder/image%206.png)

![image.png](standart_autoencoder/image%207.png)

![image.png](standart_autoencoder/image%208.png)

![image.png](standart_autoencoder/image%209.png)

![image.png](standart_autoencoder/image%2010.png)

![image.png](standart_autoencoder/image%2011.png)

![image.png](standart_autoencoder/image%2012.png)

============================================================
LATENT SPACE STATISTICS
============================================================
Latent vector dimension: 16384
Mean latent value: 1.078246
Std latent value: 0.920277
Min latent value: -3.047380
Max latent value: 12.810332
Sparsity (% zeros): 0.84%
============================================================

**Per-Class Reconstruction Comparison**

[https://www.notion.so](https://www.notion.so)

`================================================================================
                    AUTOENCODER EXPERIMENT - COMPREHENSIVE SUMMARY
================================================================================

[1] MODEL ARCHITECTURE
----------------------------------------
    Model Type: Convolutional Autoencoder
    Input Size: 256x256x3
    Base Channels: 32
    Latent Channels: 64
    Total Parameters: 464,739

[2] DATASET INFORMATION
----------------------------------------
    Training Samples: 76,500
    Validation Samples: 8,500
    Test Samples: 355
    Number of Classes: 17
    Classes: cm, cmm, p1, p2, p3, p31m, p3m1, p4, p4g, p4m, p6, p6m, pg, pgg, pm, pmg, pmm

[3] TRAINING CONFIGURATION
----------------------------------------
    Epochs: 30
    Batch Size: 32
    Learning Rate: 0.001
    Weight Decay: 1e-05
    Patience (Early Stopping): 5
    Optimizer: AdamW
    Scheduler: CosineAnnealingLR
    Loss Function: MSE

[4] TRAINING RESULTS
----------------------------------------
    Total Epochs Trained: 21
    Final Training Loss: 0.009903
    Final Validation Loss: 0.009579
    Best Validation Loss: 0.009285

[5] TEST RESULTS
----------------------------------------
    Test MSE Loss: 0.060758
    Test RMSE: 0.246492

[6] PER-CLASS PERFORMANCE (Top 5 Best & Worst)
----------------------------------------
    Best Reconstructed Classes:
      - p1: 0.030025 ± 0.027309
      - pm: 0.036713 ± 0.030132
      - cm: 0.043554 ± 0.033758
      - p3m1: 0.046524 ± 0.038429
      - pmg: 0.049887 ± 0.038073

    Worst Reconstructed Classes:
      - pmm: 0.068911 ± 0.051598
      - pgg: 0.074798 ± 0.081493
      - p6m: 0.087812 ± 0.075404
      - p4: 0.107977 ± 0.126374
      - p6: 0.110968 ± 0.096030

[7] SAVED ARTIFACTS
----------------------------------------
    Output Directory: /content/artifacts
      - feature_maps.png
      - latent_statistics.png
      - error_distribution.png
      - tsne_latent_space.png
      - training_history.png
      - reconstruction_comparison.png
      - pca_latent_space.png
      - per_class_reconstructions.png
      - feature_maps_detailed.png
      - best_autoencoder.pt

================================================================================
                              END OF SUMMARY
================================================================================`