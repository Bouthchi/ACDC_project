# ACDC Cardiac MRI Segmentation: Classical Baselines vs U-Net

We segment cardiac MRI into four classes per pixel: **background (0), right-ventricular cavity (1), myocardium (2) and left-ventricular cavity (3)**. We compare tuned classical pixel classifiers (Logistic Regression and Random Forest) with a U-Net that uses an ImageNet-pretrained ResNet-34 encoder. We also analyse where the models fail and what the improvement costs.

## Main results

Mean foreground Dice. Each score averages the two cardiac phases (ED, ES) within each patient, then averages patients equally, then averages the three structures. Scores are computed on the preprocessed 1.5 mm, 288 × 288 grid.

| Model | Validation (20 patients) | **Test (50 unseen patients)** |
|---|---:|---:|
| Logistic Regression | 0.069 | **0.073** |
| Random Forest (RF_06: 100 trees, depth 20, leaf 2) | 0.347 | **0.340** |
| Original U-Net (ResNet-34 encoder) | 0.890 | 0.901 |
| **Final model: RV-weighted U-Net + post-processing** | **0.893** | **0.898** |
| RV-weighted U-Net, fine-tuned with augmentation + post-processing | 0.895 | 0.902 |

Final model on the test set by structure: **RV 0.894, myocardium 0.877, LV 0.924**.

**Findings**
- The U-Net scores about 2.6× higher than the tuned Random Forest on unseen patients. The classical models over-segment: they have low precision and high recall.
- The cost: U-Net training took about 21–23 min on a GPU, against about 3 min of CPU fitting for the forest. The U-Net has 24.4 M parameters.
- Validation and test scores agree closely, so the 20-patient validation split did not give a misleading picture.
- RV weighting, fine-tuning and post-processing all land within ±0.004 Dice of each other on test. None of them produced a clear, reliable improvement.
- The final model was selected on validation with a rule fixed in advance. It was **not** changed after seeing test results.

**Failure analysis**
- End-systole (contracted heart) is harder than end-diastole: test mean Dice is 0.881 vs 0.916. The gap is largest for the RV.
- Hypertrophic cardiomyopathy (HCM) is the weakest disease group (0.879). Four of the five worst test volumes are HCM at end-systole, where the LV cavity becomes very small.
- The errors concentrate in basal and apical slices.
- Averaging predictions over flipped images collapsed RV Dice (to about 0.15–0.27) for models trained without flips. This suggests the CNN relies on the RV's usual position in the image. The Random Forest shows the same reliance: its most important feature is row position.

## Download the final model

The trained weights (93 MB) are attached to the **[v1.0 release](../../releases)**.

```python
import torch, segmentation_models_pytorch as smp

model = smp.Unet('resnet34', encoder_weights=None, in_channels=1, classes=4)
ckpt = torch.load('acdc_unet_resnet34_rv_weighted_final.pth', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
# Input: tensor (N, 1, 288, 288), one MRI slice per item, preprocessed as described below.
# Output: (N, 4, 288, 288) class scores -> argmax over dim 1 gives labels 0-3.
# Post-processing used for the final model: keep the largest 3D connected component per structure.
```

## Method summary

| Step | What we did |
|---|---|
| Data | ACDC: 100 development patients plus 50 official test patients, each with end-diastole and end-systole volumes and expert masks |
| Split | Patient-level 80/20 split of the development patients, stratified by the five conditions, seed 42. The test patients were used once, at the end. |
| Preprocessing | In-plane resampling to 1.5 mm (linear for images, nearest-neighbour for masks), centre crop/pad to 288 × 288, z-score of nonzero voxels per volume. No annotated tissue was lost by cropping. |
| Classical models | Five pixel features (intensity, 5 × 5 local mean and std, normalised row and column) and class-balanced pixel sampling. Logistic Regression on standardised features; Random Forest with 8 configurations tuned on validation. |
| U-Net | `segmentation_models_pytorch` U-Net with an ImageNet ResNet-34 encoder, 1 input channel and 4 output classes, all layers trained. AdamW (encoder LR 1e-4, decoder LR 1e-3), loss = cross-entropy + foreground Dice, 20 epochs, checkpoint chosen by validation loss. |
| Mitigation | The RV fails at end-systole, so we doubled the RV weight in the Dice loss. We then fine-tuned with augmentation (rotation, scaling, flips, contrast; 10 epochs, 10× lower LR) and tested largest-component post-processing. |
| Selection rule | Highest validation mean Dice. Candidates within 0.002 of the best are separated by end-systole RV Dice. |

## Repository contents

| File | What it is |
|---|---|
| `ACDC_Cardiac_Segmentation.ipynb` | Full code with all outputs: data checks, preprocessing, models, evaluation and figures |
| `acdc_results.zip` | All result figures, test and validation score tables (summary and per volume), the final model choice, the fine-tuning settings and the patient split |
| `final_model_selection_validation.csv` | Validation scores used to choose the final model |
| `final_model_validation_per_volume.csv` | Final model's validation scores for each volume |
| `finetune_history.csv` | Validation scores after each fine-tuning epoch |

The trained model is attached to the **v1.0 release** (see above).

## Reproducing

1. Download ACDC from the official site (see Data below) and put the patient ZIPs in your own Google Drive with this layout:
   ```
   MyDrive/ACDC dataset/ACDC_patient_downloads/
       patient001.zip ... patient100.zip
       testing/patient101.zip ... patient150.zip
   ```
2. Open the notebook in Google Colab with a **T4 GPU** runtime (Runtime → Change runtime type).
3. Run the cells in order. The notebook creates the patient split, the preprocessed data, all models and all results in that same Drive folder. Training and evaluation cells reuse saved results if they already exist.

One known gap: Section 7.2 reads the original U-Net's validation scores from a CSV, and the cell that first wrote that CSV is not in the notebook. On a fresh run, that cell needs the CSV to exist. The same scores are recomputed from the saved checkpoint in Section 9.1 (0.890435, matching the saved 0.890434), so the reported numbers are still reproducible.

Software used: Python 3, PyTorch 2.14, segmentation-models-pytorch 0.6.0, scikit-learn 1.6.1, nibabel, SciPy, NumPy, pandas, Matplotlib.

The MRI data is **not** included in this repository. It must be obtained from the official ACDC source.

## Limitations

- One training run per U-Net configuration with a single seed, so the small differences between U-Net variants are within run-to-run variation.
- Learning rates were fixed standard transfer-learning values, not searched.
- 2D slice model, with no context between slices. Dice is measured on the resampled grid, not in the original image space.
- Model confidence (calibration) was not evaluated.

## Data

ACDC dataset: O. Bernard et al., "Deep Learning Techniques for Automatic MRI Cardiac Multi-Structures Segmentation and Diagnosis: Is the Problem Solved?", *IEEE Transactions on Medical Imaging*, 37(11), 2018. https://www.creatis.insa-lyon.fr/Challenge/acdc/
