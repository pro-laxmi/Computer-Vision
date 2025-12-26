import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
from io import BytesIO
import requests

# ==================== MODEL CLASSES ====================
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, embed_dim=128, patch_size=7):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=128, attention_heads=8, dropout=0.1, mlp_dim=256):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim, attention_heads, dropout, bias=True)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        x = x.transpose(0, 1)
        residual_1 = x
        x = self.layer_norm_1(x)
        x = self.multi_head_attention(x, x, x)[0] + residual_1
        residual_2 = x
        x = self.layer_norm_2(x)
        x = self.mlp(x) + residual_2
        x = x.transpose(0, 1)
        return x


class MLPhead(nn.Module):
    def __init__(self, embed_dim=128, mlp_dim=256):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        x = self.layer_norm_1(x)
        x = self.mlp_head(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, num_classes=10,
                 embed_dim=128, depth=6, num_heads=8, mlp_dim=256, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.transformer_blocks = nn.Sequential(*[TransformerEncoder(embed_dim, num_heads, dropout, mlp_dim) for _ in range(depth)])
        self.mlp_head = MLPhead(embed_dim, mlp_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer_blocks(x)
        x = x[:, 0]
        x = self.mlp_head(x)
        return x


@st.cache_resource
def load_model(device):
    """Load the trained model"""
    model_config = {
        'img_size': 28,
        'patch_size': 7,
        'in_channels': 1,
        'num_classes': 10,
        'embed_dim': 128,
        'depth': 6,
        'num_heads': 8,
        'mlp_dim': 256,
        'dropout': 0.1
    }
    
    model = VisionTransformer(**model_config).to(device)
    
    model_path = './trained_vit_mnist.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.warning("⚠️ Model file not found. Please train the model first!")
    
    model.eval()
    return model


def predict_digit(image, model, device):
    """Predict the digit in the image"""
    # Resize image to 28x28
    img_resized = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to tensor
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = 1.0 - img_array  # Invert colors
    
    # Create tensor with batch dimension
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0, prediction].item() * 100
        all_probs = probabilities[0].cpu().numpy() * 100
    
    return prediction, confidence, all_probs


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
    st.markdown('<div class="subtitle">Powered by Vision Transformer</div>', unsafe_allow_html=True)
    
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
        ### 🔬 Vision Transformer Architecture
        
        This application uses a **Vision Transformer (ViT)** model trained on the MNIST dataset. 
        The model converts images into patches and processes them using transformer-based attention mechanisms.
        
        **Model Architecture:**
        - Patch Size: 7×7
        - Embedding Dimension: 128
        - Attention Heads: 8
        - Transformer Layers: 6
        - Dataset: MNIST (60,000 training images)
        
        ### 📊 Model Performance
        - Trained on MNIST handwritten digits (0-9)
        - Input image size: 28×28 pixels
        - Works best with clear, centered digits
        
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
