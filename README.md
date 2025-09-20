# OWLSAM: Zero-Shot Object Detection + 2D Segmentation

**OWLSAM** combines state-of-the-art models for object detection and segmentation:

- **OWL-V2**: Zero-shot object detection — detect objects without retraining for specific labels.
- **SAM (Segment Anything Model)**: High-quality 2D mask generation for detected objects.

This integration allows you to input an image along with **candidate labels** (comma-separated) and get **pixel-accurate masks** for the detected objects.

---

## 🚀 Features

- Zero-shot detection: No need to train for custom objects.
- Pixel-perfect segmentation masks with SAM.
- Simple interface via Gradio (web-based or Colab).

---

## 💻 How to Run

### Locally

1. Clone the repository:

```bash
git clone https://github.com/yourusername/OWLSAM-Colab.git
cd OWLSAM-Colab
