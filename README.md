# 🔢 MNIST Digit Recognizer - Vision Transformer Web App

A beautiful, interactive web application for recognizing handwritten digits using a Vision Transformer model trained on MNIST dataset.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🧠 Model Architecture

### Vision Transformer Model Architecture

```
INPUT: 28×28 Grayscale Image
       │
       ▼
┌────────────────────────────┐
│  Patch Embedding              │
│  - Conv2d Layer               │
│  - Kernel: 7×7                │
│  - Stride: 7                  │
│  - Output: (1, 16, 128)       │
│    (1 batch, 16 patches,      │
│     128 embedding dims)       │
└────────────────────────────┘
       │
       ▼
┌────────────────────────────┐
│  Add CLS Token                │
│  - Prepend learnable token    │
│  - Shape: (1, 17, 128)        │
└────────────────────────────┘
       │
       ▼
┌────────────────────────────┐
│  Add Positional Embedding     │
│  - Learnable positional       │
│    embeddings                 │
│  - Shape: (1, 17, 128)        │
└────────────────────────────┘
       │
       ▼
  ┌────────────────────────┐
  │  TRANSFORMER BLOCK  ×6   │
  │  ┌──────────────────┐  │
  │  │ Layer Norm         │  │
  │  └────────┬─────────┘  │
  │            ▼             │
  │  ┌──────────────────┐  │
  │  │ Multi-Head Attn    │  │
  │  │ (8 heads)          │  │
  │  └────────┬─────────┘  │
  │            ▼             │
  │  ┌──────────────────┐  │
  │  │ Add Residual       │  │
  │  └────────┬─────────┘  │
  │            ▼             │
  │  ┌──────────────────┐  │
  │  │ Layer Norm         │  │
  │  └────────┬─────────┘  │
  │            ▼             │
  │  ┌──────────────────┐  │
  │  │ MLP (FC+ReLU+FC)   │  │
  │  └────────┬─────────┘  │
  │            ▼             │
  │  ┌──────────────────┐  │
  │  │ Add Residual       │  │
  │  └────────┬─────────┘  │
  │            ▼             │
  │  Output: (1, 17, 128)    │
  └────────────────────────┘
       │
       ▼
┌────────────────────────────┐
│  Extract CLS Token            │
│  - Take first token only      │
│  - Shape: (1, 128)            │
└────────────────────────────┘
       │
       ▼
┌────────────────────────────┐
│  MLP Head                     │
│  - Layer Norm                 │
│  - FC: 128 → 256              │
│  - FC: 256 → 128              │
│  - Shape: (1, 128)            │
└────────────────────────────┘
       │
       ▼
OUTPUT: Logits for 10 digits
        Shape: (1, 10)
```

---

## 🎯 Demo

### Features:
1. **Draw Tab** - Sketch digits with your mouse
2. **Upload Tab** - Upload digit images from files
3. **About Tab** - Learn about the model and tips

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip package manager
- Trained model file: `trained_vit_mnist.pth`

### Installation

1. **Clone or download the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/mnist-digit-recognizer.git
   cd mnist-digit-recognizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_web.txt
   ```

3. **Make sure you have the trained model**
   ```bash
   # The trained_vit_mnist.pth should be in the same directory
   ls trained_vit_mnist.pth
   ```

4. **Run the app**
   ```bash
   streamlit run vit_web_app.py
   ```

5. **Access the app**
   - Open your browser to: `http://localhost:8501`

---

## 📦 Installation Details

### For Windows Users:
```powershell
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install packages
pip install -r requirements_web.txt

# Run app
streamlit run vit_web_app.py
```

### For Mac/Linux Users:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements_web.txt

# Run app
streamlit run vit_web_app.py
```

---

## 🏗️ Project Structure

```
mnist-digit-recognizer/
├── vit_web_app.py                 # Main Streamlit app
├── vit_drawing_gui.py             # Desktop GUI version
├── trained_vit_mnist.pth          # Trained model weights
├── requirements_web.txt           # Web app dependencies
├── DEPLOYMENT_GUIDE.md            # Detailed deployment guide
├── README.md                       # This file
├── .streamlit/
│   └── config.toml               # Streamlit configuration
└── CNN_and_Vision/06_vision_transformer_from_scratch.ipynb  # Training notebook
```

---


## 💡 Usage Tips

### For Best Results:
1. **Draw clearly** - Use contrasting colors (dark on light)
2. **Center the digit** - Place digit in the middle of canvas
3. **Full size** - Make the digit reasonably large
4. **Single digit** - Only one digit per image
5. **Clear image** - Avoid smudges or multiple strokes

### Example:
- Good: Clear, centered digit in a 28×28 MNIST style
- Bad: Multiple digits, very small, at edges, faint/light

---

## 🛠️ Customization

### Modify Appearance:
Edit the custom CSS in `vit_web_app.py`:
```python
st.markdown("""
<style>
.main-title {
    color: #YOUR_COLOR;
    font-size: 3em;
}
</style>
""", unsafe_allow_html=True)
```

### Change Model Hyperparameters:
Edit the `model_config` in `vit_web_app.py`:
```python
model_config = {
    'img_size': 28,
    'patch_size': 7,
    'embed_dim': 128,  # Change this
    'depth': 6,
    # ... other params
}
```

---

## 📊 Performance

### Inference Time:
- **CPU**: ~100-200ms
- **GPU (CUDA)**: ~20-50ms

### Accuracy:
- Model trained to achieve ~97-99% accuracy on MNIST test set

### App Response:
- Drawing prediction: <1 second
- Upload prediction: 1-2 seconds

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:**
```bash
pip install streamlit
```

### Issue: "trained_vit_mnist.pth not found"
**Solution:**
- Train the model first using the Jupyter notebook
- Or download from project releases
- Make sure file is in the same directory as `vit_web_app.py`

### Issue: App is very slow
**Solution:**
- First load takes 10-30 seconds on free tier
- Subsequent requests are faster
- GPU deployment is faster

### Issue: Drawing canvas not working
**Solution:**
```bash
pip install streamlit-drawable-canvas
```

### Issue: "CUDA out of memory"
**Solution:**
```python
# In vit_web_app.py, change:
device = torch.device('cpu')  # Force CPU
```

---

## 📚 Learning Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **Vision Transformer Paper**: https://arxiv.org/abs/2010.11929
- **MNIST Dataset**: http://yann.lecun.com/exdb/mnist/
- **PyTorch Docs**: https://pytorch.org/docs/
- **Vizuara Playlist**: https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgmWYoSpY_2EJzPJjkke4Az

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:
- Add batch prediction
- Implement webcam input
- Add more datasets (CIFAR-10, etc.)
- Improve UI/UX
- Add model comparison
- Performance optimizations

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

Created with ❤️ by Laxmidhar Panda.

---

## 🙏 Acknowledgments

- Original MNIST dataset by Yann LeCun
- Vision Transformer paper by Google Research
- Vizuara for it's awesome and indeapth video leactures on this
- Streamlit for amazing ML web framework

---

**Enjoy tinkering! ✨**