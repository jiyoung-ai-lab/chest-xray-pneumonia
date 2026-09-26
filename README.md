# 🩻 Chest X-Ray Pneumonia

**Chest X-ray pneumonia classification using PyTorch and Transfer Learning**

> Exploring model selection, fine-tuning, generalization, and model error analysis.

---

## 📌 Progress

| Area                             | Status |
| -------------------------------- | ------ |
| Dataset Analysis & Preprocessing | ✅      |
| ResNet50 Transfer Learning       | ✅      |
| DenseNet121 Transfer Learning    | ✅      |
| Learning Rate Comparison         | ✅      |
| Fine-Tuning Experiments          | ✅      |
| Model Comparison                 | ✅      |
| Test Evaluation                  | ✅      |
| Error Analysis & Grad-CAM        | ✅      |
| Experiment Tracking              | ✅      |

---

## 📊 Results

| Model                  | Validation Accuracy | Test Accuracy |
| ---------------------- | ------------------: | ------------: |
| **DenseNet121 · Full** |          **99.62%** |    **81.25%** |
| **ResNet50 · Full**    |          **99.33%** |    **75.96%** |

> **Key finding:** High validation performance did not directly translate to independent test performance.

---

## 🔍 Project Highlights

* Compared **ResNet50 / DenseNet121** with different fine-tuning strategies
* Evaluated **generalization** using an independent test set
* Applied **Grad-CAM** for qualitative error analysis
* Identified possible reliance on **non-diagnostic image features**
* Designed experiments around **reproducibility and experiment tracking**

---

## 📁 Project Structure

```text
chest-xray-pneumonia/
│
├── cxp/                 # Project modules
├── notebooks/           # Experiments & analysis
├── experiments/         # Run configurations & model artifacts
└── reports/             # Experiment metadata & results
```

---

## 🚀 Next Step

**Preprocessing & Generalization**

* Annotation / Text Masking
* Lung ROI / Segmentation
* Preprocessing comparison
* Error Analysis & Grad-CAM re-evaluation

---

> 🚧 **Currently in progress**
