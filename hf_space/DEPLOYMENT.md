# 🚀 Hugging Face Spaces Deployment Guide

## 📦 What's in this folder?

This is a **lightweight version** of the Movie Recommendation System optimized for Hugging Face Spaces (under 50GB limit).

### Files:
- `app.py` - Gradio interface with collaborative filtering (no PyTorch)
- `requirements.txt` - Minimal dependencies (gradio, pandas, numpy, scikit-learn)
- `README.md` - Hugging Face Spaces metadata
- `data/movies.csv` - MovieLens movie dataset
- `data/ratings.csv` - MovieLens ratings dataset

**Total Size: ~2.4MB** ✅

## 🔧 Changes from full version:

1. **Removed PyTorch models** - No heavy neural networks
2. **Traditional collaborative filtering** - Uses pandas/numpy instead
3. **No model training** - Lightweight recommendation logic only
4. **Minimal dependencies** - No torch, matplotlib, tqdm
5. **Sample data fallback** - Works even if CSVs aren't loaded

## 📤 How to deploy:

### Option 1: Via Hugging Face Web UI

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose:
   - Space name: `movie-recommendation-system`
   - SDK: **Gradio**
   - Hardware: **CPU basic** (free tier)
4. Upload all files from this `hf_space/` folder
5. Wait for the build to complete
6. Your app will be live! 🎉

### Option 2: Via Git (recommended)

```bash
# Clone your HF Space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/movie-recommendation-system
cd movie-recommendation-system

# Copy files from hf_space folder
cp -r ../hf_space/* .

# Push to HF Spaces
git add .
git commit -m "Initial deployment of lightweight movie recommender"
git push
```

### Option 3: Via Hugging Face CLI

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload the space
huggingface-cli upload-space YOUR_USERNAME/movie-recommendation-system ./hf_space
```

## ✅ Pre-deployment checklist:

- [x] Total size < 50GB (current: 2.4MB)
- [x] No large .pth model files
- [x] No mlruns/ or logs/ directories
- [x] Minimal requirements.txt
- [x] README.md with proper metadata
- [x] Sample data fallback implemented
- [x] Gradio interface tested locally

## 🧪 Test locally before deploying:

```bash
cd hf_space
pip install -r requirements.txt
python app.py
```

Open http://localhost:7860 to test the app.

## 🔍 What was excluded to save space?

From the main project, we excluded:
- `mlruns/` - MLflow tracking artifacts (~GB)
- `logs/` - Training logs
- `weights/` - Pre-trained PyTorch models (4 x ~50MB each)
- `*.pth` files - Model checkpoints
- `outputs/` - Experiment results
- Training scripts (`train.py`, `train_hydra.py`)
- Full model implementations (PyTorch neural nets)
- Multiple deployment folders
- Test data and notebooks

## 💡 Performance notes:

- **Cold start**: ~5-10 seconds (Gradio + pandas loading)
- **Recommendation speed**: <1 second for most queries
- **Memory usage**: ~200-300MB RAM
- **Works on**: CPU basic tier (free)

## 🆘 Troubleshooting:

### "Application startup failed"
- Check the build logs in HF Spaces
- Ensure `requirements.txt` matches app imports
- Verify data files uploaded correctly

### "Storage limit exceeded"
- Confirm you're uploading ONLY files from `hf_space/` folder
- Don't upload the parent `reco_app/` folder
- Size should be <5MB

### "Import errors"
- Make sure requirements.txt is present
- Check Python version compatibility (3.9+)

## 🎯 Next steps after deployment:

1. Test all tabs (Recommendations, Profile, Search, Trending)
2. Try different user IDs (1-610)
3. Share your Space URL!
4. Optional: Enable GPU if you want to add PyTorch models later

## 📊 Comparison with full version:

| Feature | Full Version | HF Spaces Version |
|---------|--------------|-------------------|
| Size | ~5GB+ | 2.4MB |
| Dependencies | PyTorch, MLflow | Pandas, NumPy |
| Training | ✅ 4 models | ❌ (too heavy) |
| Recommendations | ✅ Neural networks | ✅ Collaborative filtering |
| Speed | Medium | Fast |
| Hardware | GPU recommended | CPU sufficient |

---

**Ready to deploy?** Follow Option 1 or 2 above! 🚀
