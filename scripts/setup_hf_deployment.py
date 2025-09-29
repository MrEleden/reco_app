#!/usr/bin/env python3
"""
Setup script for Hugging Face Spaces deployment
"""
import os
import shutil
from pathlib import Path


def setup_hf_deployment():
    """Prepare files for Hugging Face Spaces deployment."""

    print("🚀 Setting up Hugging Face Spaces deployment...")

    # Create deployment directory
    deploy_dir = Path("hf_deploy")
    deploy_dir.mkdir(exist_ok=True)

    print(f"📁 Created deployment directory: {deploy_dir}")

    # Copy essential files
    files_to_copy = [
        ("apps/app.py", "app.py"),
        ("README_huggingface.md", "README.md"),
        ("deployment/requirements_hf.txt", "requirements.txt"),
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

    print("\n🎉 Deployment setup complete!")
    print(f"\n📋 Next steps:")
    print(f"1. Go to https://huggingface.co/new-space")
    print(f"2. Create a new Streamlit space")
    print(f"3. Upload all files from '{deploy_dir}' folder")
    print(f"4. Your app will be live at: https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME")

    return deploy_dir


if __name__ == "__main__":
    setup_hf_deployment()
