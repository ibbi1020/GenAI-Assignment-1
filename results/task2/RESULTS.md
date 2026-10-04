# Task 2 results — hard-routed restoration

A CNN classifier looks at the corrupted photo and names the damage: clean, salt and pepper, Gaussian blur, or rectangular occlusion. The photo is then sent to one specialist autoencoder for that damage. Clean photos skip restoration (identity bypass).

Code: `src/genai/training/task2_classifier.py`, `task2_specialists.py`, `routing.py`. Scripts: `scripts/run_task2_classifier.py`, `run_task2_specialists.py`, `evaluate_task2.py`.

## Classifier

Optuna ran 10 trials of 5 epochs each, with no pruner, on validation macro-F1. Trial 0 (the starting guess) won with 0.9752. The next best was trial 2 at 0.9751, so the choice is close.

| Setting | Value |
| --- | --- |
| Learning rate | 1e-3 |
| Batch size | 32 |
| Base channels | 32 |
| Blocks | 4 |
| Dropout | 0.1 |
| Weight decay | 1e-4 |

The final run stopped early at epoch 12. Final validation macro-F1 was 0.9861.

Test split (3,669 photos × 10 conditions = 36,690 images):

| Accuracy | Macro-F1 | Macro precision | Macro recall |
| --- | --- | --- | --- |
| 0.9776 | 0.9648 | 0.9560 | 0.9766 |

| Class | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Clean | 0.835 | 0.972 | 0.898 |
| Salt and pepper | 1.000 | 0.999 | 1.000 |
| Gaussian blur | 0.991 | 0.999 | 0.995 |
| Rectangular occlusion | 0.999 | 0.936 | 0.967 |

Accuracy by severity:

| Corruption | Low | Medium | High |
| --- | --- | --- | --- |
| Salt and pepper | 99.7% | 100% | 100% |
| Gaussian blur | 99.8% | 100% | 100% |
| Rectangular occlusion | **81.5%** | 99.5% | 100% |
| Clean | 97.2% | | |

The one weak spot is low-severity occlusion. Most of those errors are called clean. Clean precision is low (0.835) because those images land in the clean bucket. Files: `classifier_summary.json`, `confusion_matrix.png/.csv`, `classifier_optuna_trials.json`.

## Specialists

Three autoencoders (same `DenoisingAutoencoder` class as Task 1), each trained only on its own corruption. One shared search picked the settings for all three.

The search was planned for 6 trials of 3 epochs. It was **stopped by hand during trial 5**, so 5 trials finished. Score is the mean validation `L1 + (1 - SSIM)` over the three specialists. Lower is better.

| Trial | Score | Bottleneck | Channels | Norm | Alpha |
| --- | --- | --- | --- | --- | --- |
| 0 | **0.316** | 32 | 64 | group | 0.5 |
| 1 | 0.384 | 16 | 32 | batch | 0.87 |
| 2 | 0.499 | 16 | 32 | batch | 0.85 |
| 3 | 0.477 | 16 | 64 | group | 0.81 |
| 4 | 0.413 | 16 | 128 | batch | 0.85 |
| 5 | stopped | | | | |

Trial 0 was the starting guess and no later trial beat it. Three epochs favours fast-learning settings, so this says little about how the other settings would do with more training. Final settings: lr 5e-4, batch 32, bottleneck 32, 64 channels, group norm, alpha 0.5.

Each specialist was then trained for up to 15 epochs (patience 5). All three used the full 15 epochs and were still improving.

| Specialist | Best validation score |
| --- | --- |
| Salt and pepper | 0.1892 |
| Gaussian blur | 0.1674 |
| Rectangular occlusion | 0.2546 |

ONNX files match PyTorch (largest difference `6.1e-6`). The search log is `logs/task2_specialists_search.log` and the final training log is `logs/task2_specialists_final.log`.

## Routing on the test split

Oracle routing uses the true label. Predicted routing uses the classifier. 2.24% of images were misrouted.

| System | L1 | SSIM | PSNR |
| --- | --- | --- | --- |
| Do nothing | 0.0506 | 0.695 | 25.3 |
| Task 1 autoencoder | 0.0381 | 0.806 | 25.3 |
| Oracle routing | 0.0352 | 0.843 | 30.8 |
| **Predicted routing** | **0.0351** | **0.843** | **30.5** |

Per corruption (mean of the three severities), L1 / SSIM / PSNR:

| Corruption | Do nothing | Task 1 | Predicted routing |
| --- | --- | --- | --- |
| Clean | 0 / 1.000 / 80 | 0.0233 / 0.927 / 30.1 | 0.0010 / 0.997 / 78.5 |
| Salt and pepper | 0.0433 / 0.412 / 16.4 | 0.0391 / 0.762 / 25.5 | **0.0325 / 0.842 / 27.1** |
| Gaussian blur | **0.0287 / 0.829 / 27.8** | 0.0319 / 0.828 / 27.2 | 0.0331 / 0.836 / 27.0 |
| Rectangular occlusion | 0.0965 / 0.743 / 13.5 | **0.0481** / 0.789 / **21.6** | 0.0509 / **0.800** / **21.6** |

What this shows:

- Predicted routing is almost identical to oracle routing, so classifier errors cost very little overall.
- Salt and pepper: the specialist is clearly the best.
- Occlusion: routing and Task 1 are about equal. Routing has better SSIM and slightly worse L1.
- **Blur: neither approach beats the corrupted input** on L1 or PSNR. SSIM is slightly higher for routing (0.836 vs 0.829). By severity, the loss comes from low blur: routed L1 0.0287 / SSIM 0.898 / PSNR 28.3 against input 0.0157 / 0.948 / 32.3. At medium and high blur routing helps on SSIM and PSNR (medium SSIM 0.865 vs 0.821, PSNR 27.7 vs 26.8; high SSIM 0.746 vs 0.717, PSNR 24.9 vs 24.4) and L1 is about equal. On validation the blur specialist looked marginally better than the input; on test it does not.
- Clean photos are nearly untouched under routing, while the Task 1 model degrades them.

Files: `routing_summary.json`, `oracle_test_metrics.csv/.json`, `predicted_test_metrics.csv/.json`, `identity_test_metrics.csv`, `universal_test_metrics.csv`, `routing_failures.json/.png`, `predicted_examples.png`.

## Known issues to state in the report

- Blur restoration does not improve on the corrupted input. More specialist epochs may help; the models were still improving at epoch 15.
- The specialist search was short (3 epochs) and was stopped one trial early.
- The latent in every specialist at bottleneck 32 and 64 channels is 32×32×128 = 131,072 values, 2.7 times the input size. It is smaller than Task 1's latent but still not a compression.
- Low-severity occlusion is often called clean.
