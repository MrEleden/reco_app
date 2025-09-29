#!/usr/bin/env python3
"""
Setup script for Gradio Hugging Face Spaces deployment
"""
import os
import shutil
from pathlib import Path


def setup_gradio_deployment():
    """Prepare files for Gradio Hugging Face Spaces deployment."""

    print("🎬 Setting up Gradio Hugging Face Spaces deployment...")

    # Create deployment directory
    deploy_dir = Path("hf_gradio_deploy")
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir(exist_ok=True)

    print(f"📁 Created deployment directory: {deploy_dir}")

    # Copy essential files for Gradio
    files_to_copy = [
        ("apps/app_gradio.py", "app.py"),  # Gradio expects app.py
        ("README_gradio.md", "README.md"),
        ("deployment/requirements_gradio.txt", "requirements.txt"),
    ]

    for src, dst in files_to_copy:
        if Path(src).exists():
            shutil.copy2(src, deploy_dir / dst)
            print(f"✅ Copied {src} → {dst}")
        else:
            print(f"⚠️ Missing {src}")

    # Copy data directory
    data_src = Path("data")
    data_dst = deploy_dir / "data"
    if data_src.exists():
        shutil.copytree(data_src, data_dst, dirs_exist_ok=True)
        print(f"✅ Copied {data_src} → {data_dst}")

    # Copy weights directory
    weights_src = Path("weights")
    weights_dst = deploy_dir / "weights"
    if weights_src.exists():
        shutil.copytree(weights_src, weights_dst, dirs_exist_ok=True)
        print(f"✅ Copied {weights_src} → {weights_dst}")

    # Copy Python modules required for MLflow model loading
    modules_to_copy = ["models", "data", "losses", "metrics", "optimizers", "utils"]

    for module in modules_to_copy:
        module_src = Path(module)
        module_dst = deploy_dir / module
        if module_src.exists():
            shutil.copytree(module_src, module_dst, dirs_exist_ok=True)
            print(f"✅ Copied {module_src} → {module_dst}")
        else:
            print(f"⚠️ Missing module {module_src}")

    # Copy mlruns directory (for MLflow)
    mlruns_src = Path("mlruns")
    mlruns_dst = deploy_dir / "mlruns"
    if mlruns_src.exists():
        print("📊 Copying MLflow experiments...")
        shutil.copytree(mlruns_src, mlruns_dst, dirs_exist_ok=True)
        print(f"✅ Copied {mlruns_src} → {mlruns_dst}")
    else:
        print("⚠️ MLflow runs directory not found - some features may not work")

    print("\n🎉 Gradio deployment setup complete!")
    print(f"\n📋 Next steps:")
    print(f"1. Go to https://huggingface.co/new-space")
    print(f"2. Choose 'Gradio' SDK")
    print(f"3. Create a new space")
    print(f"4. Upload all files from '{deploy_dir}' folder")
    print(f"5. Your Gradio app will be live!")

    print(f"\n✨ Gradio Features:")
    print(f"   🤖 Interactive model selection")
    print(f"   🎬 Real-time movie recommendations")
    print(f"   📊 Model performance comparison")
    print(f"   🎯 Clean, ML-focused interface")

    return deploy_dir


if __name__ == "__main__":
    setup_gradio_deployment()
