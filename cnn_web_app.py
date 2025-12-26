import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os
from io import BytesIO
import cv2

# ==================== MODEL CLASSES ====================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x


@st.cache_resource
def load_model(device):
    """Load the trained CNN model"""
    model = CNN().to(device)
    
    model_path = './trained_CNN_mnist.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.warning("⚠️ Model file not found. Please train the model first!")
    
    model.eval()
    return model


def predict_digit(image, model, device):
    """
    MNIST-faithful preprocessing for CNN inference
    """

    # 1. Convert to grayscale
    image = image.convert("L")

    # 2. Convert to numpy
    img = np.array(image, dtype=np.uint8)

    # 3. Invert if background is white
    if img.mean() > 127:
        img = 255 - img

    # 4. Threshold (remove anti-aliasing & noise)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Find bounding box of digit
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        return None, 0.0, None  # empty canvas

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    img = img[y0:y1, x0:x1]

    # 6. Make image square by padding
    h, w = img.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img

    # 7. Resize to 28x28 (MNIST style)
    img = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

    # 8. Convert to float and normalize
    img = img.astype(np.float32) / 255.0

    # MNIST normalization
    img = (img - 0.1307) / 0.3081

    # 9. Convert to tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    # 10. Inference
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0, pred].item() * 100
        all_probs = probs[0].cpu().numpy() * 100

    return pred, confidence, all_probs



# ==================== STREAMLIT APP ====================
def main():
    # Set page config
    st.set_page_config(
        page_title="MNIST Digit Recognizer",
        page_icon="🔢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .prediction-box {
        border-radius: 10px;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2em;
        font-weight: bold;
        margin: 20px 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-title">🔢 MNIST Digit Recognizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Powered by Convolutional Neural Network (CNN)</div>', unsafe_allow_html=True)
    
    # Setup device and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["✏️ Draw Digit", "📸 Upload Image", "ℹ️ About"])
    
    # ==================== TAB 1: DRAWING ====================
    with tab1:
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.subheader("Draw a Digit")
            st.write("Use the canvas below to draw a digit (0-9). Draw clearly in the center for best results.")
            
            # Use streamlit-drawable-canvas
            try:
                from streamlit_drawable_canvas import st_canvas
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 1)",
                    stroke_width=15,
                    stroke_color="rgba(0, 0, 0, 1)",
                    background_color="rgba(255, 255, 255, 1)",
                    height=300,
                    width=300,
                    drawing_mode="freedraw",
                    key="canvas",
                )
                
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                predict_btn = col_btn1.button("🔮 Predict", use_container_width=True, key="predict_canvas")
                clear_btn = col_btn2.button("🧹 Clear", use_container_width=True, key="clear_canvas")
                
                if clear_btn:
                    st.rerun()
                
                if predict_btn and canvas_result.image_data is not None:
                    # Get image from canvas
                    canvas_image = canvas_result.image_data.astype(np.uint8)
                    pil_image = Image.fromarray(canvas_image, mode='RGBA').convert('L')
                    
                    # Predict
                    prediction, confidence, all_probs = predict_digit(pil_image, model, device)
                    
                    # Display results
                    with col2:
                        st.subheader("Prediction Results")
                        st.markdown(f'<div class="prediction-box">{prediction}</div>', unsafe_allow_html=True)
                        
                        st.metric("Confidence", f"{confidence:.2f}%")
                        
                        st.write("**All Probabilities:**")
                        prob_df = {}
                        for i, prob in enumerate(all_probs):
                            prob_df[f"Digit {i}"] = f"{prob:.2f}%"
                        
                        # Create a nicer visualization
                        col_prob1, col_prob2 = st.columns(2)
                        for i in range(5):
                            with col_prob1:
                                st.progress(float(all_probs[i]) / 100, text=f"Digit {i}: {all_probs[i]:.1f}%")
                        for i in range(5, 10):
                            with col_prob2:
                                st.progress(float(all_probs[i]) / 100, text=f"Digit {i}: {all_probs[i]:.1f}%")
                
            except ImportError:
                st.error("⚠️ Please install streamlit-drawable-canvas: `pip install streamlit-drawable-canvas`")
    
    # ==================== TAB 2: UPLOAD ====================
    with tab2:
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.subheader("Upload Handwritten Digit")
            st.write("Upload an image of a handwritten digit (works best with 28x28 images)")
            
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file).convert('L')
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("🔮 Predict", use_container_width=True, key="predict_upload"):
                    # Predict
                    prediction, confidence, all_probs = predict_digit(image, model, device)
                    
                    # Display results
                    with col2:
                        st.subheader("Prediction Results")
                        st.markdown(f'<div class="prediction-box">{prediction}</div>', unsafe_allow_html=True)
                        
                        st.metric("Confidence", f"{confidence:.2f}%")
                        
                        st.write("**All Probabilities:**")
                        for i in range(5):
                            st.progress(float(all_probs[i]) / 100, text=f"Digit {i}: {all_probs[i]:.1f}%")
                        for i in range(5, 10):
                            st.progress(float(all_probs[i]) / 100, text=f"Digit {i}: {all_probs[i]:.1f}%")
        
        with col2:
            st.subheader("Sample MNIST Digits")
            st.info("💡 Tip: Upload images similar to MNIST dataset for best accuracy")
    
    # ==================== TAB 3: ABOUT ====================
    with tab3:
        st.subheader("About This Application")
        
        st.write("""
        ### 🔬 Convolutional Neural Network (CNN) Architecture
        
        This application uses a **CNN** model trained on the MNIST dataset. 
        The model uses convolutional layers to extract spatial features and fully connected layers for classification.
        
        **Model Architecture:**
        - **Conv Layer 1:** 1 input channel → 10 filters (5×5 kernel) + Max Pooling (2×2)
        - **Conv Layer 2:** 10 filters → 20 filters (5×5 kernel) + Dropout + Max Pooling (2×2)
        - **Dense Layer 1:** 320 units → 50 units (ReLU activation)
        - **Dense Layer 2:** 50 units → 10 units (output logits)
        - **Dataset:** MNIST (60,000 training images, 10,000 test images)
        - **Optimizer:** Adam
        - **Loss Function:** Cross Entropy Loss
        
        ### 📊 Model Performance
        - Trained on MNIST handwritten digits (0-9)
        - Input image size: 28×28 pixels (grayscale)
        - Works best with clear, centered digits
        - Typical accuracy: 98%+
        
        ### 🎯 How to Use
        1. **Draw Tab**: Use your mouse to draw a digit in the canvas
        2. **Upload Tab**: Upload a handwritten digit image
        3. The model will automatically predict the digit and show confidence scores
        
        ### ⚙️ Technical Details
        - Framework: PyTorch
        - Interface: Streamlit
        - Device: CPU/GPU (auto-detected)
        
        ### 📝 Tips for Best Results
        - Draw digits clearly and centered
        - Use dark ink on white background
        - Ensure digits are roughly the size of MNIST samples
        - Avoid multiple digits in one image
        """)
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Device", str(device).upper())
        with col2:
            st.metric("Model Status", "✅ Loaded" if model else "❌ Not Loaded")
        with col3:
            st.metric("Input Size", "28×28")


if __name__ == "__main__":
    main()
