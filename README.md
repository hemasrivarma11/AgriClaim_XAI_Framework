# AgriClaim-XAI: Multi-Agent Crop Damage Assessment & Insurance Framework

This repository implements **AgriClaim-XAI**, an explainable, multi-agent, multimodal framework designed for intelligent crop damage assessment, context consistency verification, and automated crop insurance claim reasoning.

---

## 1. Prerequisites & Installation

Ensure you have Python 3.10+ and standard PyTorch installed. Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 2. Stepwise Execution Commands

Follow these steps to test, evaluate, and launch the application:

### Step 1: Run Quick CLI Integration Tests
To verify the entire 7-agent pipeline (Data Integration, Swin Vision, Context Verification with growth stages, Severity Agent, Claim Reasoning Agent, Explainability Agent, and Report Generator), run:
```bash
python run.py --test
```

### Step 2: Run Inference on a Single Crop Photo
To process a custom image through the multi-agent pipeline and generate detailed Officer/Farmer reports:
```bash
python run.py --image samples/rice_leaf_blight.png --crop Rice --season Monsoon --damage blight
```

### Step 3: Run the Complete Experimental Benchmark Suite
To download/simulate datasets (PlantVillage & PlantDoc), train baseline models, compute significance p-values, execute ablation studies, and render 300 DPI figures and multi-format tables:
```bash
python 03_Code/experimental_suite.py
```

### Step 4: Launch the Interactive Streamlit Dashboard
To launch the interactive web dashboard for real-time claim upload, context validation overrides, and SHAP/Grad-CAM visualizations:
```bash
streamlit run app.py
```

---

## 3. Output Folder Structure

After executing the benchmark evaluation suite (Step 3), the outputs will be organized under the `outputs/` directory:

```text
outputs/
├── models/      # Trained model weights (.pth checkpoints)
├── results/     # CSV evaluation tables (accuracy, regression, t-test p-values)
├── tables/      # Publication-ready tables in CSV, Excel (.xlsx), and Markdown (.md)
├── figures/     # High-resolution (300 DPI) Matplotlib curves, matrices, and SHAP plots
├── reports/     # Detailed Academic Discussion report (evaluation_discussion.txt)
└── logs/        # Execution logs
```
