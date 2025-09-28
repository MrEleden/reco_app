"""
MLflow Server Manager
====================

This script helps you manage the MLflow server and ensures it stays running
and up-to-date with your experiments.
"""

import subprocess
import sys
import time
import os
from pathlib import Path


def check_mlflow_server():
    """Check if MLflow server is running on port 5000."""
    try:
        import requests

        response = requests.get("http://127.0.0.1:5000", timeout=2)
        return response.status_code == 200
    except:
        return False


def start_mlflow_server():
    """Start MLflow server with proper configuration."""

    print("🚀 Starting MLflow Server...")
    print("-" * 40)

    # Ensure mlruns directory exists
    mlruns_path = Path("mlruns").resolve()
    mlruns_path.mkdir(exist_ok=True)

    # MLflow command
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
        "--backend-store-uri",
        f"file:///{mlruns_path}",
    ]

    print(f"📂 Backend store: {mlruns_path}")
    print(f"🌐 Server URL: http://127.0.0.1:5000")
    print(f"⚡ Command: {' '.join(cmd)}")
    print()

    try:
        # Start server
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=os.getcwd())

        # Wait a moment for server to start
        time.sleep(3)

        # Check if server started successfully
        if check_mlflow_server():
            print("✅ MLflow server started successfully!")
            print("📊 Open: http://127.0.0.1:5000")
            print("⏹️  Press Ctrl+C to stop the server")
            print()

            # Keep server running and show logs
            try:
                while True:
                    line = process.stdout.readline()
                    if line:
                        print(f"[MLflow] {line.strip()}")
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping MLflow server...")
                process.terminate()
                process.wait()
                print("✅ Server stopped.")
        else:
            print("❌ Failed to start MLflow server")
            stdout, stderr = process.communicate()
            if stderr:
                print(f"Error: {stderr}")

    except Exception as e:
        print(f"❌ Error starting server: {e}")


def main():
    """Main function."""

    print("MLflow Server Manager")
    print("=" * 30)

    # Check if server is already running
    if check_mlflow_server():
        print("✅ MLflow server is already running!")
        print("📊 Open: http://127.0.0.1:5000")
        print("💡 If you need to restart, press Ctrl+C and run this script again")
        return

    # Start server
    start_mlflow_server()


if __name__ == "__main__":
    main()
