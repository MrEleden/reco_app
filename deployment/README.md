# Deployment Directory

Contains deployment configurations and ready-to-deploy packages for different platforms.

## 📁 Structure

```
deployment/
├── requirements_gradio.txt      # Gradio-specific dependencies
├── requirements_hf.txt          # Hugging Face optimized requirements  
├── requirements_streamlit.txt   # Streamlit-specific dependencies
├── hf_deploy/                   # Streamlit → HF Spaces package
├── hf_gradio_deploy/           # Gradio → HF Spaces package (recommended)
└── README.md                   # This file
```

## 🚀 Deployment Options

### Option 1: Gradio on Hugging Face Spaces ⭐ **RECOMMENDED**
```
Use: hf_gradio_deploy/
Platform: Hugging Face Spaces
SDK: Gradio
Features: Clean ML demo interface
```

### Option 2: Streamlit on Hugging Face Spaces  
```
Use: hf_deploy/
Platform: Hugging Face Spaces  
SDK: Docker (with Streamlit)
Features: Full-featured dashboard
```

## 📋 Deployment Steps

### For Gradio (Recommended):
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **"Gradio"** SDK
3. Upload all files from `hf_gradio_deploy/`
4. Your app goes live automatically!

### For Streamlit:
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)  
2. Choose **"Docker"** SDK
3. Upload all files from `hf_deploy/`
4. Requires Docker configuration

## 🔧 Requirements Files

- **requirements_gradio.txt**: Minimal dependencies for Gradio demo
- **requirements_streamlit.txt**: Full dependencies for Streamlit app
- **requirements_hf.txt**: Optimized for Hugging Face deployment

## ✨ Features Comparison

| Feature | Gradio Package | Streamlit Package |
|---------|----------------|-------------------|
| **Setup** | One-click | Docker config needed |
| **Interface** | Clean, ML-focused | Rich, customizable |
| **Performance** | Lightweight | Feature-rich |
| **Best for** | Demos, sharing | Development, analysis |

Choose **Gradio** for easiest deployment and professional ML demos!