import torch
import torch.nn as nn
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
import os

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


# ==================== GUI APPLICATION ====================
class DrawingApp:
    def __init__(self, root, model, device):
        self.root = root
        self.root.title("MNIST Digit Drawing - Vision Transformer")
        self.root.geometry("600x700")
        
        self.model = model
        self.device = device
        
        # Drawing canvas parameters
        self.canvas_size = 280
        self.brush_size = 15
        self.drawing = False
        
        # Create PIL image for drawing
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
        
        # Title
        title_label = tk.Label(root, text="Draw a Digit (0-9)", font=("Arial", 20, "bold"))
        title_label.pack(pady=10)
        
        # Instructions
        instructions = tk.Label(root, text="Draw a digit in the white canvas. The model will predict it automatically.",
                               font=("Arial", 10), fg="gray")
        instructions.pack(pady=5)
        
        # Create canvas
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size,
                               bg='white', cursor="cross", relief=tk.SUNKEN, borderwidth=2)
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)
        
        # Result frame
        result_frame = tk.Frame(root)
        result_frame.pack(pady=10)
        
        # Prediction result
        tk.Label(result_frame, text="Prediction:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        self.result_label = tk.Label(result_frame, text="?", font=("Arial", 48, "bold"), fg="red")
        self.result_label.pack(side=tk.LEFT, padx=10)
        
        # Confidence
        self.confidence_label = tk.Label(root, text="Confidence: --", font=("Arial", 10))
        self.confidence_label.pack()
        
        # Buttons frame
        button_frame = tk.Frame(root)
        button_frame.pack(pady=15)
        
        tk.Button(button_frame, text="Clear", command=self.clear_canvas, width=12, height=2).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Predict", command=self.predict, width=12, height=2).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Exit", command=root.quit, width=12, height=2).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_label = tk.Label(root, text=f"Device: {device}", font=("Arial", 9), fg="blue")
        self.status_label.pack(side=tk.BOTTOM, pady=5)
    
    def start_draw(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw_line(self, event):
        if self.drawing:
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                   fill='black', width=self.brush_size, capstyle=tk.ROUND, smooth=True)
            
            # Draw on PIL image (scale down from canvas to actual image size)
            scale = self.canvas_size / self.canvas_size
            pil_x1 = int(self.last_x * scale)
            pil_y1 = int(self.last_y * scale)
            pil_x2 = int(event.x * scale)
            pil_y2 = int(event.y * scale)
            
            self.draw.line([pil_x1, pil_y1, pil_x2, pil_y2], fill=0, width=self.brush_size)
            
            self.last_x = event.x
            self.last_y = event.y
    
    def end_draw(self, event):
        self.drawing = False
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="?")
        self.confidence_label.config(text="Confidence: --")
    
    def predict(self):
        # Resize image to 28x28 (MNIST size)
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        # Invert colors (white background to black, black drawing to white)
        img_array = 1.0 - img_array
        
        # Create tensor with batch dimension
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[0, prediction].item() * 100
        
        # Update labels
        self.result_label.config(text=str(prediction))
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")


def load_model(model_path, device):
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
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}")
        print("Using untrained model. Please train the model first in the notebook.")
    
    model.eval()
    return model


if __name__ == "__main__":
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = './trained_vit_mnist.pth'
    model = load_model(model_path, device)
    
    # Create GUI
    root = tk.Tk()
    app = DrawingApp(root, model, device)
    root.mainloop()
