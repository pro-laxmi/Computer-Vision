import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
import os
import cv2
import io
import time

# ==============================================================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==============================================================================

st.set_page_config(
    page_title="NeuroDigit | Dual-Core MNIST Recognizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism, Animations, and Layout
st.markdown("""
<style>
    /* REDUCE TOP WHITESPACE/PADDING */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* MAIN CONTAINER STYLES */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* TYPOGRAPHY */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2c3e50;
    }
    
    .main-header {
        text-align: center;
        padding: 0.5rem 0 1rem 0; /* Reduced padding */
        animation: fadeInDown 1s ease-out;
    }
    
    .main-title {
        background: -webkit-linear-gradient(45deg, #4b6cb7, #182848);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 900;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 0;
    }
    
    .subtitle {
        color: #555;
        font-size: 1.25rem;
        font-weight: 300;
        letter-spacing: 1px;
        margin-top: -10px;
    }

    /* CARDS & CONTAINERS */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 24px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.12);
    }

    /* PREDICTION BOX */
    .prediction-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    .big-pred-digit {
        font-size: 6rem;
        font-weight: 800;
        color: white;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        width: 150px;
        height: 150px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 10px 20px rgba(118, 75, 162, 0.4);
        animation: popIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        margin-bottom: 1rem;
    }

    /* MODEL BADGES */
    .model-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .badge-cnn { background-color: #e3f2fd; color: #1565c0; border: 1px solid #bbdefb; }
    .badge-vit { background-color: #f3e5f5; color: #7b1fa2; border: 1px solid #e1bee7; }

    /* ANIMATIONS */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.5); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* CUSTOM TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    /* FOOTER */
    .footer {
        text-align: center;
        color: #888;
        padding: 2rem;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. MODEL ARCHITECTURES
# ==============================================================================

# ----------------------------
# 2.1 Convolutional Neural Network (CNN)
# ----------------------------
class CNN(nn.Module):
    """
    A classic Convolutional Neural Network for MNIST digit classification.
    Structure: Conv -> MaxPool -> Conv -> Dropout -> MaxPool -> FC -> FC
    """
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional Layer 1: 1 input channel (grayscale), 10 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # Convolutional Layer 2: 10 input channels, 20 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Dropout regularization to prevent overfitting
        self.conv2_drop = nn.Dropout2d()
        # Fully Connected Layer 1: 320 input features -> 50 output features
        self.fc1 = nn.Linear(320, 50)
        # Fully Connected Layer 2 (Output): 50 input features -> 10 classes (digits 0-9)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Layer 1: Conv -> MaxPool(2x2) -> ReLU
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Layer 2: Conv -> Dropout -> MaxPool(2x2) -> ReLU
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten: Reshape tensor for the fully connected layer
        x = x.view(-1, 320)
        # Layer 3: FC -> ReLU
        x = F.relu(self.fc1(x))
        # Dropout during training
        x = F.dropout(x, p=0.5, training=self.training)
        # Layer 4: Output Layer
        x = self.fc2(x)
        return x

# ----------------------------
# 2.2 Vision Transformer (ViT)
# ----------------------------
class PatchEmbedding(nn.Module):
    """Splits image into patches and embeds them."""
    def __init__(self, in_channels=1, embed_dim=128, patch_size=7):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        # Use Conv2d to implement patch projection efficiently
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, E, H', W']
        x = x.flatten(2)  # [B, E, N]
        x = x.transpose(1, 2)  # [B, N, E]
        return x

class TransformerEncoder(nn.Module):
    """Single Block of the Transformer Encoder."""
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
        # Attention Block with Residual Connection
        x = x.transpose(0, 1) # MHA expects [Seq, Batch, Embed]
        residual_1 = x
        x = self.layer_norm_1(x)
        x, _ = self.multi_head_attention(x, x, x)
        x = x + residual_1
        
        # MLP Block with Residual Connection
        residual_2 = x
        x = self.layer_norm_2(x)
        x = self.mlp(x) + residual_2
        x = x.transpose(0, 1) # Return to [Batch, Seq, Embed]
        return x

class MLPhead(nn.Module):
    """Classification Head for ViT."""
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
    """
    A lightweight Vision Transformer (ViT) designed for MNIST.
    """
    def __init__(self, img_size=28, patch_size=7, in_channels=1, num_classes=10,
                 embed_dim=128, depth=6, num_heads=8, mlp_dim=256, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        
        # Learnable Class Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Learnable Positional Embedding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        # Transformer Blocks
        self.transformer_blocks = nn.Sequential(*[
            TransformerEncoder(embed_dim, num_heads, dropout, mlp_dim) for _ in range(depth)
        ])
        
        # Output Head
        self.mlp_head = MLPhead(embed_dim, mlp_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        
        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Positional Embedding
        x = x + self.pos_embed
        
        # Pass through Transformer
        x = self.transformer_blocks(x)
        
        # Use the CLS token output for classification
        x = x[:, 0]
        x = self.mlp_head(x)
        return x


# ==============================================================================
# 3. UTILITY FUNCTIONS (LOADING & PREPROCESSING)
# ==============================================================================

@st.cache_resource
def get_device():
    """Detects if CUDA is available, otherwise uses CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model(model_type, device):
    """
    Factory function to load the specific model architecture.
    """
    if model_type == "CNN":
        model = CNN().to(device)
        path = './trained_CNN_mnist.pth'
    else:
        # ViT Hyperparameters
        model_config = {
            'img_size': 28, 'patch_size': 7, 'in_channels': 1, 'num_classes': 10,
            'embed_dim': 128, 'depth': 6, 'num_heads': 8, 'mlp_dim': 256, 'dropout': 0.1
        }
        model = VisionTransformer(**model_config).to(device)
        path = './trained_vit_mnist.pth'
    
    # Attempt to load weights
    weights_loaded = False
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            weights_loaded = True
        except Exception as e:
            st.error(f"Error loading weights: {e}")
    
    model.eval()
    return model, weights_loaded, path

def preprocess_image(image):
    """
    Robust Preprocessing Pipeline for MNIST Inference.
    Handles canvas drawings and uploaded images.
    
    Steps:
    1. Grayscale
    2. Resize to intermediate size
    3. Invert (if needed)
    4. Denoise
    5. Threshold
    6. Center via Bounding Box
    7. Square Pad
    8. Resize to 28x28
    9. Normalize
    """
    # 1. Convert to grayscale
    if image.mode != "L":
        image = image.convert("L")
    
    # Convert to numpy
    img_array = np.array(image)
    
    # 2. Invert logic: MNIST is White on Black. 
    # If the image is mostly light (mean > 127), assume it's Black on White and invert.
    if img_array.mean() > 127:
        img_array = 255 - img_array
        
    # 3. Denoise (Gaussian Blur) - helps smooth out pixelated canvas drawings
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
    
    # 4. Thresholding (Otsu's Binarization) - creates crisp digits
    _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Find Bounding Box (Centering)
    coords = cv2.findNonZero(img_array)
    if coords is None:
        return None # Empty canvas
        
    x, y, w, h = cv2.boundingRect(coords)
    
    # Crop to the digit
    digit_crop = img_array[y:y+h, x:x+w]
    
    # 6. Square Padding
    # We want the digit to fit in a 20x20 box inside the 28x28 image (like MNIST)
    max_dim = max(w, h)
    
    # Create a square canvas of the max dimension + padding
    pad_factor = 1.4 # Add 40% padding around the digit
    square_size = int(max_dim * pad_factor)
    
    # Create black square background
    square_img = np.zeros((square_size, square_size), dtype=np.uint8)
    
    # Paste digit in center
    y_pos = (square_size - h) // 2
    x_pos = (square_size - w) // 2
    square_img[y_pos:y_pos+h, x_pos:x_pos+w] = digit_crop
    
    # 7. Resize to 28x28
    final_img = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 8. Normalize (0-1 range then standard MNIST normalization)
    tensor_img = final_img.astype(np.float32) / 255.0
    tensor_img = (tensor_img - 0.1307) / 0.3081
    
    return tensor_img, final_img

# ==============================================================================
# 4. MAIN APPLICATION LOGIC
# ==============================================================================

def main():
    
    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png", caption="MNIST Dataset Samples")
        st.markdown("### ⚙️ Model Configuration")
        
        # Model Selector
        model_choice = st.radio(
            "Select Architecture:",
            ["CNN", "ViT"],
            captions=["Convolutional Neural Network", "Vision Transformer"],
            index=0,
            help="Choose between the classic CNN approach or the modern Transformer approach."
        )
        
        st.markdown("---")
        
        # Device Info
        device = get_device()
        st.info(f"**Computing Device:** {str(device).upper()}")
        
        # Load Model
        model, weights_loaded, model_path = load_model(model_choice, device)
        
        if weights_loaded:
            st.success(f"**{model_choice}** Weights Loaded Successfully!")
        else:
            st.warning(f"⚠️ **{model_choice}** Weights Not Found.\n(`{model_path}`)\n\nRunning with random initialization (predictions will be random).")

        st.markdown("---")
        st.markdown("### 👨‍💻 About")
        st.caption("Integrated NeuroDigit System v2.0")

    # --- MAIN CONTENT HEADER ---
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown('<div class="main-title">NeuroDigit</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Next-Generation Handwritten Digit Recognition Powered by <span style="font-weight:bold; color:#4b6cb7;">{model_choice}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- TABS ---
    tab_draw, tab_upload, tab_about = st.tabs(["🎨 **Interactive Canvas**", "📤 **Upload Image**", "📚 **Deep Dive & Architecture**"])

    # --------------------------------------------------------------------------
    # TAB 1: DRAWING CANVAS
    # --------------------------------------------------------------------------
    with tab_draw:
        col_canvas, col_pred = st.columns([1.2, 1], gap="large")
        
        with col_canvas:
            st.subheader("Draw a Digit")
            st.write("Draw a digit (0-9) inside the box. Use the tools below to correct mistakes.")
            
            # Canvas Component
            try:
                from streamlit_drawable_canvas import st_canvas
                
                # Center the canvas using columns
                # 300px canvas on a wide screen needs centering in its column
                left_pad, canvas_col, right_pad = st.columns([1, 4, 1])
                
                with canvas_col:
                    # Canvas settings
                    stroke_width = st.slider("Stroke Width", 10, 30, 20)
                    
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 255, 255, 1)",
                        stroke_width=stroke_width,
                        stroke_color="#FFFFFF", # Drawing in white on black logic inverted later, or black on white
                        background_color="#000000", # Black background is native for MNIST
                        height=300,
                        width=300,
                        drawing_mode="freedraw",
                        key="canvas",
                        display_toolbar=True # Enables Undo/Redo/Trash
                    )
                
                # Instructions
                st.caption("Tip: Draw large and centered for best results. The model expects white text on black background (handled automatically).")
                
            except ImportError:
                st.error("Please install `streamlit-drawable-canvas`")

        with col_pred:
            st.subheader("Real-time Analysis")
            
            if canvas_result.image_data is not None:
                # Process the canvas data
                # Canvas returns RGBA. 
                img_data = canvas_result.image_data.astype(np.uint8)
                pil_image = Image.fromarray(img_data)
                
                # Preprocess
                processed_data = preprocess_image(pil_image)
                
                if processed_data is not None:
                    tensor_img, debug_img = processed_data
                    
                    # Add batch dimensions and send to device
                    input_tensor = torch.from_numpy(tensor_img).unsqueeze(0).unsqueeze(0).to(device)
                    
                    # Inference
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)
                        conf, pred_class = torch.max(probs, 1)
                        
                        confidence = conf.item() * 100
                        prediction = pred_class.item()
                        all_probs = probs[0].cpu().numpy() * 100
                    
                    # --- DISPLAY RESULTS ---
                    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
                    st.markdown(f'<div class="model-badge badge-{"cnn" if model_choice=="CNN" else "vit"}">{model_choice} Model</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="big-pred-digit">{prediction}</div>', unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** {confidence:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Probability Bars
                    st.write("**Class Probabilities:**")
                    
                    # Top 3 classes
                    top3_idx = np.argsort(all_probs)[-3:][::-1]
                    for idx in top3_idx:
                        st.progress(int(all_probs[idx]), text=f"Digit {idx}: {all_probs[idx]:.1f}%")
                    
                    # Expander for full debug view
                    with st.expander("🛠️ See Model Input (Debug)"):
                        st.image(debug_img, caption="Processed Image (28x28)", width=100)
                        st.caption("This is exactly what the model sees after cropping, resizing, and normalizing.")
                else:
                    st.info("Waiting for input... Draw something on the left!")

    # --------------------------------------------------------------------------
    # TAB 2: UPLOAD
    # --------------------------------------------------------------------------
    with tab_upload:
        col_up_1, col_up_2 = st.columns([1, 1], gap="large")
        
        with col_up_1:
            st.subheader("Upload an Image")
            uploaded_file = st.file_uploader("Choose a JPG or PNG file", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Upload", width=250)

        with col_up_2:
            st.subheader("Analysis Results")
            
            if uploaded_file is not None:
                if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
                    with st.spinner(f"Running {model_choice} Inference..."):
                        time.sleep(0.5) # UX feel
                        
                        processed_data = preprocess_image(image)
                        
                        if processed_data is not None:
                            tensor_img, debug_img = processed_data
                            input_tensor = torch.from_numpy(tensor_img).unsqueeze(0).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                output = model(input_tensor)
                                probs = torch.softmax(output, dim=1)
                                conf, pred_class = torch.max(probs, 1)
                            
                            prediction = pred_class.item()
                            confidence = conf.item() * 100
                            all_probs = probs[0].cpu().numpy() * 100
                            
                            # Result Display
                            col_res_a, col_res_b = st.columns([1, 2])
                            with col_res_a:
                                st.image(debug_img, caption="Input to Model", width=100)
                            with col_res_b:
                                st.markdown(f"### Prediction: **{prediction}**")
                                st.markdown(f"Confidence: **{confidence:.2f}%**")
                            
                            st.bar_chart(all_probs)
                        else:
                            st.error("Could not process image. Is it empty?")
            else:
                st.info("Please upload an image to begin.")

    # --------------------------------------------------------------------------
    # TAB 3: EXTENSIVE ABOUT SECTION (THE "LARGE" PART)
    # --------------------------------------------------------------------------
    with tab_about:
        st.markdown("## 🧠 Comprehensive Architecture Analysis")
        st.markdown("""
        This application demonstrates the evolution of Computer Vision by allowing you to switch between a classical **Convolutional Neural Network (CNN)** and a state-of-the-art **Vision Transformer (ViT)**. Both models were trained on the MNIST dataset (60,000 training images).
        """)

        # --- SECTION 1: CNN ---
        st.markdown("### 1. Convolutional Neural Network (CNN)")
        with st.expander("Show CNN Architecture Details", expanded=False):
            st.markdown("""
            **Philosophy:** Inductive Bias. CNNs assume that pixels close to each other are related (locality) and that patterns are the same regardless of where they appear (translation invariance).
            
            #### 🏗️ Architecture Blueprint
            The model consists of two main blocks followed by a classifier.
            
            1.  **Feature Extraction Block 1**
                * **Conv2d:** Input 1ch → Output 10ch. Kernel $5\\times5$.
                * **MaxPool2d:** Kernel $2\\times2$. Reduces spatial dimension from $28\\times28$ to $12\\times12$.
                * **ReLU:** Activation function $f(x) = \max(0, x)$.
            
            2.  **Feature Extraction Block 2**
                * **Conv2d:** Input 10ch → Output 20ch. Kernel $5\\times5$.
                * **Dropout:** $p=0.5$. Randomly zeroes out neurons to prevent memorization.
                * **MaxPool2d:** Kernel $2\\times2$. Reduces dimension to $4\\times4$.
                * **ReLU:** Activation.
            
            3.  **Classification Head**
                * **Flatten:** $20 \ channels \\times 4 \\times 4 = 320$ features.
                * **Linear (Dense):** $320 \\rightarrow 50$.
                * **Linear (Output):** $50 \\rightarrow 10$ (Logits for digits 0-9).
            
            #### 🧮 Mathematical Context
            The convolution operation is defined as:
            $$ (f * g)(t) = \int_{-\infty}^{\infty} f(\\tau)g(t - \\tau) d\\tau $$
            In discrete image terms, we slide a filter $K$ over image $I$:
            $$ S(i, j) = (I * K)(i, j) = \sum_m \sum_n I(m, n) K(i-m, j-n) $$
            """)

        # --- SECTION 2: VIT ---
        st.markdown("### 2. Vision Transformer (ViT)")
        with st.expander("Show ViT Architecture Details", expanded=False):
            st.markdown("""
            **Philosophy:** Global Attention. Unlike CNNs, ViTs do not assume locality bias strictly. They treat an image as a sequence of patches (like words in a sentence) and learn relationships between any two patches regardless of distance using Self-Attention.
            
            #### 🏗️ Architecture Blueprint
            The model adapts the Transformer from NLP for Vision.
            
            1.  **Patch Embedding**
                * Image ($28\\times28$) is split into fixed-size patches ($7\\times7$).
                * Number of patches: $(28/7) \\times (28/7) = 4 \\times 4 = 16$ patches.
                * Each patch is flattened and projected linearly to an `embed_dim` of 128.
            
            2.  **Positional Encoding**
                * Since the Transformer has no notion of order/space, we add learnable vectors to the patch embeddings to indicate position.
                * Includes a special **[CLS] token** (Class Token) prepended to the sequence.
            
            3.  **Transformer Encoder Blocks (x6)**
                * **Layer Norm:** Stabilizes training.
                * **Multi-Head Self Attention (MSA):** The core mechanism.
                    $$ \text{Attention}(Q, K, V) = \text{softmax}(\\frac{QK^T}{\sqrt{d_k}})V $$
                * **MLP:** A small feed-forward network inside the block.
            
            4.  **MLP Head**
                * Takes the output of the **[CLS] token** only.
                * Projects $128 \\rightarrow 256 \\rightarrow 10$ classes.
            """)

        # --- SECTION 3: PREPROCESSING ---
        st.markdown("### 3. The Vision Pipeline")
        with st.expander("How we process your drawing", expanded=True):
            cols_pipe = st.columns(4)
            with cols_pipe[0]:
                st.markdown("**1. Input**")
                st.caption("Raw Canvas RGBA")
            with cols_pipe[1]:
                st.markdown("**2. Cleanup**")
                st.caption("Grayscale → Denoise → Threshold")
            with cols_pipe[2]:
                st.markdown("**3. Geometry**")
                st.caption("Center of Mass → Square Pad")
            with cols_pipe[3]:
                st.markdown("**4. Tensor**")
                st.caption("Resize 28x28 → Normalize")
            
            st.code("""
# Python Code for the Pipeline used in this app:
def preprocess(image):
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    x, y, w, h = cv2.boundingRect(img)
    # ... Square Padding Logic ...
    img = cv2.resize(img, (28, 28))
    img = (img - 0.1307) / 0.3081 # MNIST Mean/Std
    return img
            """, language="python")

    # --- FOOTER ---
    st.markdown('<div class="footer">Built with Streamlit, PyTorch, and OpenCV • Merging Classic & Modern AI</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()