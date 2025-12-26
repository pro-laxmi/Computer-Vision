#!/usr/bin/env python3
"""
Quick test script to verify the web app setup
Run this before deploying
"""

import sys
import os

def check_requirements():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...\n")
    
    requirements = {
        'streamlit': 'Streamlit',
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'streamlit_drawable_canvas': 'Streamlit Drawable Canvas'
    }
    
    missing = []
    for module, name in requirements.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall missing packages with:")
        print("pip install -r requirements_web.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True

def check_model_file():
    """Check if trained model exists"""
    print("\n🔍 Checking model file...\n")
    
    model_path = './trained_vit_mnist.pth'
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model found: {model_path}")
        print(f"   Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ Model not found: {model_path}")
        print("\nTo fix:")
        print("1. Train the model using: 06_vision_transformer_from_scratch.ipynb")
        print("2. Or download the model from the project repository")
        return False

def check_app_file():
    """Check if web app file exists"""
    print("\n🔍 Checking web app file...\n")
    
    app_path = './vit_web_app.py'
    
    if os.path.exists(app_path):
        print(f"✅ Web app found: {app_path}")
        return True
    else:
        print(f"❌ Web app not found: {app_path}")
        return False

def print_summary(results):
    """Print test summary"""
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    if all(results.values()):
        print("✅ All checks passed!")
        print("\n🚀 Ready to run the app:")
        print("   streamlit run vit_web_app.py")
        print("\n🌐 Ready to deploy to Streamlit Cloud!")
        return True
    else:
        print("❌ Some checks failed.")
        print("\nFailing checks:")
        for check, result in results.items():
            if not result:
                print(f"  - {check}")
        return False

def main():
    print("="*50)
    print("🧪 MNIST Digit Recognizer - Setup Test")
    print("="*50)
    print()
    
    results = {
        'Dependencies': check_requirements(),
        'Model File': check_model_file(),
        'Web App File': check_app_file(),
    }
    
    success = print_summary(results)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
