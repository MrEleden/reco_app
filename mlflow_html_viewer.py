"""
MLflow Web Alternative - Clean Experiment Viewer
===============================================

If MLflow UI has rendering issues, use this as a clean alternative
to view your experiments in a simple web interface.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.mlflow_utils import MLflowModelSelector
import json
from datetime import datetime


def generate_html_report():
    """Generate a clean HTML report of experiments."""

    try:
        selector = MLflowModelSelector(experiment_name="movie_recommendation")
        comparison = selector.compare_models()

        if comparison.empty:
            return "<html><body><h1>No experiments found</h1></body></html>"

        # Generate HTML
        html = (
            """
<!DOCTYPE html>
<html>
<head>
    <title>MLflow Experiments - Movie Recommendation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { color: #1f77b4; margin-bottom: 30px; }
        .best-model { background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        .rank-1 { background-color: #fff3cd; }
        .rank-2 { background-color: #f8f9fa; }
        .status-finished { color: #28a745; font-weight: bold; }
        .metric { font-family: monospace; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 Movie Recommendation System - Experiments</h1>
        <p>Generated: """
            + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            + """</p>
    </div>
"""
        )

        # Best model summary
        best_model = comparison.iloc[0]
        html += f"""
    <div class="best-model">
        <h2>🏆 Best Model Performance</h2>
        <p><strong>Model:</strong> {best_model['model_type']}</p>
        <p><strong>RMSE:</strong> <span class="metric">{best_model['val_rmse']:.4f}</span></p>
        <p><strong>Accuracy:</strong> <span class="metric">{best_model['val_accuracy']:.4f}</span></p>
        <p><strong>F1-Score:</strong> <span class="metric">{best_model['val_f1']:.4f}</span></p>
        <p><strong>Run ID:</strong> <span class="metric">{best_model['run_id']}</span></p>
    </div>
"""

        # Experiments table
        html += """
    <h2>📊 All Experiments</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Model Type</th>
            <th>RMSE</th>
            <th>Accuracy</th>
            <th>F1-Score</th>
            <th>Status</th>
            <th>Run ID</th>
        </tr>
"""

        for idx, run in comparison.iterrows():
            rank_class = "rank-1" if idx == 0 else "rank-2" if idx == 1 else ""
            status_class = "status-finished" if run["status"] == "FINISHED" else ""

            html += f"""
        <tr class="{rank_class}">
            <td>{idx + 1}</td>
            <td><strong>{run['model_type']}</strong></td>
            <td class="metric">{run['val_rmse']:.4f}</td>
            <td class="metric">{run['val_accuracy']:.4f}</td>
            <td class="metric">{run['val_f1']:.4f}</td>
            <td class="{status_class}">{run['status']}</td>
            <td class="metric">{run['run_id'][:8]}...</td>
        </tr>
"""

        html += """
    </table>
    
    <div style="margin-top: 30px;">
        <h2>💡 Quick Actions</h2>
        <ul>
            <li><strong>MLflow UI:</strong> <a href="http://127.0.0.1:5001" target="_blank">http://127.0.0.1:5001</a></li>
            <li><strong>Run new experiment:</strong> <code>python train_hydra.py model=hybrid</code></li>
            <li><strong>Compare models:</strong> <code>python train_hydra.py -m model=collaborative,hybrid</code></li>
            <li><strong>Check status:</strong> <code>python check_mlflow.py</code></li>
        </ul>
    </div>
    
</body>
</html>
"""

        return html

    except Exception as e:
        return f"""
<html><body>
<h1>Error generating report</h1>
<p>Error: {str(e)}</p>
<p>Try running: <code>python check_mlflow.py</code></p>
</body></html>
"""


def main():
    """Generate and save HTML report."""

    print("🌐 Generating clean HTML experiment report...")

    html_content = generate_html_report()

    # Save HTML file
    report_file = "mlflow_experiments_report.html"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Report generated: {report_file}")
    print(f"📂 Open in browser: file:///{os.path.abspath(report_file)}")

    # Also show quick summary
    try:
        selector = MLflowModelSelector(experiment_name="movie_recommendation")
        comparison = selector.compare_models()

        if not comparison.empty:
            best = comparison.iloc[0]
            print(f"\n🏆 Best Model: {best['model_type']} (RMSE: {best['val_rmse']:.4f})")
            print(f"📊 Total Experiments: {len(comparison)}")

    except Exception as e:
        print(f"⚠️  Could not load experiment summary: {e}")


if __name__ == "__main__":
    main()
