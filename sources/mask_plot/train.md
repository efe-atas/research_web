{
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
  "mask_ratio": 0.25,
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
}



Maskeleme fonksiyonlari yuklendi.
Mask ratio: 0.25 (Goruntunun 25%'i maskelenecek)



 Training samples: 76500
Validation samples: 8500
Test samples: 355
Number of classes: 17
Class names: ['cm', 'cmm', 'p1', 'p2', 'p3', 'p31m', 'p3m1', 'p4', 'p4g', 'p4m', 'p6', 'p6m', 'pg', 'pgg', 'pm', 'pmg', 'pmm']

Maskeleme ornegi (mask_ratio=0.25):



Model parameter count: 464,739
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
)


 Epoch 1: train_loss=0.070556, val_loss=0.053357
Epoch 2: train_loss=0.052917, val_loss=0.047902
Epoch 3: train_loss=0.048875, val_loss=0.047517
Epoch 4: train_loss=0.047774, val_loss=0.046998
Epoch 5: train_loss=0.046292, val_loss=0.045262
Epoch 6: train_loss=0.045175, val_loss=0.049023
Epoch 7: train_loss=0.044692, val_loss=0.044808
Epoch 8: train_loss=0.044549, val_loss=0.043000
Epoch 9: train_loss=0.043717, val_loss=0.043891
Epoch 10: train_loss=0.043100, val_loss=0.042639
Epoch 11: train_loss=0.043023, val_loss=0.043259
Epoch 12: train_loss=0.042499, val_loss=0.043013
Epoch 13: train_loss=0.042332, val_loss=0.042881
Epoch 14: train_loss=0.042005, val_loss=0.043517
Epoch 15: train_loss=0.042101, val_loss=0.041326
Epoch 16: train_loss=0.041564, val_loss=0.041814
Epoch 17: train_loss=0.041555, val_loss=0.041494
Epoch 18: train_loss=0.041311, val_loss=0.043525
Epoch 19: train_loss=0.041257, val_loss=0.041477
Epoch 20: train_loss=0.041094, val_loss=0.041058
Epoch 21: train_loss=0.040995, val_loss=0.041442
Epoch 22: train_loss=0.040891, val_loss=0.041796
Epoch 23: train_loss=0.040815, val_loss=0.041546
Epoch 24: train_loss=0.040747, val_loss=0.042233
Epoch 25: train_loss=0.040652, val_loss=0.041170
Early stopping triggered.
CPU times: user 2h 52min 45s, sys: 7min 3s, total: 2h 59min 49s
Wall time: 1h 26min 13s

 
============================================================
TRAINING SUMMARY
============================================================
Total Epochs: 25
Final Training Loss: 0.040652
Final Validation Loss: 0.041170
Best Validation Loss: 0.041058
============================================================




Test MSE Loss (Masked Inpainting): 0.065909

============================================================
CLASS-WISE RECONSTRUCTION ERROR (mean ± std)
============================================================
  p1                  : 0.033877 ± 0.029891
  pm                  : 0.041179 ± 0.034233
  cm                  : 0.047622 ± 0.038020
  p3m1                : 0.049538 ± 0.039734
  pmg                 : 0.054193 ± 0.041449
  p3                  : 0.057908 ± 0.047727
  p31m                : 0.058632 ± 0.037403
  p4m                 : 0.063517 ± 0.035886
  p2                  : 0.063687 ± 0.061649
  pg                  : 0.066616 ± 0.064319
  cmm                 : 0.068138 ± 0.080336
  p4g                 : 0.072085 ± 0.056724
  pmm                 : 0.074614 ± 0.057311
  pgg                 : 0.080547 ± 0.083709
  p6m                 : 0.094967 ± 0.081291
  p4                  : 0.113011 ± 0.128241
  p6                  : 0.118340 ± 0.103816
============================================================



PCA Analysis:
  - First 2 components explain 21.6% of variance
  - 267 components needed to explain 95% of variance







============================================================
LATENT SPACE STATISTICS
============================================================
Latent vector dimension: 16384
Mean latent value: 1.266538
Std latent value: 1.005986
Min latent value: -1.898084
Max latent value: 13.003562
Sparsity (% zeros): 0.69%
============================================================


================================================================================
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
    Total Epochs Trained: 25
    Final Training Loss: 0.040652
    Final Validation Loss: 0.041170
    Best Validation Loss: 0.041058

[5] TEST RESULTS
----------------------------------------
    Test MSE Loss: 0.065909
    Test RMSE: 0.256727

[6] PER-CLASS PERFORMANCE (Top 5 Best & Worst)
----------------------------------------
    Best Reconstructed Classes:
      - p1: 0.033877 ± 0.029891
      - pm: 0.041179 ± 0.034233
      - cm: 0.047622 ± 0.038020
      - p3m1: 0.049538 ± 0.039734
      - pmg: 0.054193 ± 0.041449

    Worst Reconstructed Classes:
      - pmm: 0.074614 ± 0.057311
      - pgg: 0.080547 ± 0.083709
      - p6m: 0.094967 ± 0.081291
      - p4: 0.113011 ± 0.128241
      - p6: 0.118340 ± 0.103816

[7] SAVED ARTIFACTS
----------------------------------------
    Output Directory: /content/artifacts
      - error_distribution.png
      - feature_maps_detailed.png
      - tsne_latent_space.png
      - pca_latent_space.png
      - training_history.png
      - per_class_reconstructions.png
      - latent_statistics.png
      - reconstruction_comparison.png
      - feature_maps.png
      - inpainting_reconstruction_comparison.png
      - best_autoencoder.pt

================================================================================
                              END OF SUMMARY
