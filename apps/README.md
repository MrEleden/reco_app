# Apps Directory

Contains the interactive applications for the movie recommendation system.

## Available Applications

### 🖥️ Streamlit App (`app.py`)
- **Purpose**: Full-featured web application with detailed visualizations
- **Features**: Model selection, performance comparison, interactive charts
- **Usage**: `streamlit run apps/app.py`
- **Best for**: Development, detailed analysis, local testing

### 🎯 Gradio App (`app_gradio.py`)  
- **Purpose**: Clean ML demo optimized for Hugging Face Spaces
- **Features**: Model selection, recommendations, comparison charts
- **Usage**: `python apps/app_gradio.py`
- **Best for**: Deployment, demos, sharing, professional showcase

## Quick Start

```bash
# Run Streamlit app
cd apps
streamlit run app.py

# Run Gradio app  
cd apps
python app_gradio.py
```

## Deployment

Both apps are ready for deployment:
- **Streamlit**: Use Docker or Streamlit Cloud
- **Gradio**: Direct deployment to Hugging Face Spaces (recommended)

See `../deployment/` folder for deployment-ready packages.