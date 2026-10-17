import os
import matplotlib.pyplot as plt
import pandas as pd

def create_workflow_diagram():
    """Generates Figure 1 / Figure 8: Complete Multi-Agent Workflow Flowchart"""
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.axis('off')
    
    # Define boxes coordinates and text
    boxes = [
        {"text": "Farmer Upload\n(Image + GPS + Metadata)", "x": 4, "y": 10.2, "color": "#e0f2fe", "border": "#0284c7"},
        {"text": "1. Data Integration Agent\n(Preprocesses & syncs inputs)", "x": 4, "y": 8.8, "color": "#f0fdf4", "border": "#16a34a"},
        {"text": "2. Crop Vision Agent\n(Swin Transformer Feature Extraction)", "x": 4, "y": 7.4, "color": "#fdf2f8", "border": "#db2777"},
        {"text": "3. Context Verification Agent\n(Validates weather and GPS consistency)", "x": 4, "y": 6.0, "color": "#fff7ed", "border": "#ea580c"},
        {"text": "4. Damage Severity Agent\n(Estimates severity & expected yield loss)", "x": 4, "y": 4.6, "color": "#fafaf9", "border": "#78716c"},
        {"text": "5. Claim Reasoning Agent\n(Fuses evidence & makes 3-tier decision)", "x": 4, "y": 3.2, "color": "#f5f3ff", "border": "#7c3aed"},
        {"text": "6. Explainability Agent\n(Grad-CAM, SHAP, Attention maps)", "x": 4, "y": 1.8, "color": "#ecfeff", "border": "#0891b2"},
        {"text": "7. Report Generation Agent\n(Compiles reports for Farmer & Officer)", "x": 4, "y": 0.4, "color": "#f0fdfa", "border": "#0d9488"},
        {"text": "Insurance Recommendation\n(Approve / Review / Reject)", "x": 4, "y": -0.8, "color": "#fee2e2", "border": "#dc2626"}
    ]
    
    # Draw boxes using bbox inside ax.text
    for box in boxes:
        ax.text(
            box["x"], box["y"], box["text"], 
            ha='center', va='center', fontsize=9, fontweight='bold', color='#1e293b',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=box["color"], edgecolor=box["border"], lw=1.5)
        )
        
    # Draw arrows
    for i in range(len(boxes) - 1):
        # We draw arrows between the text centers, matplotlib handles padding
        ax.annotate(
            "", xy=(4, boxes[i+1]["y"] + 0.4), xytext=(4, boxes[i]["y"] - 0.4),
            arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.5, mutation_scale=15)
        )
        
    ax.set_xlim(1, 7)
    ax.set_ylim(-1.5, 11)
    
    os.makedirs("04_Figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig("04_Figures/figure1_workflow.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Figure 1/8: overall workflow flowchart.")

def create_explainability_diagram():
    """Generates Figure 7: Explainability Agent Architecture"""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis('off')
    
    # Inputs on the left
    inputs = ["Visual Evidence", "Context Evidence", "Severity Evidence", "Claim Decision"]
    y_coords = [4.5, 3.3, 2.1, 0.9]
    for idx, inp in enumerate(inputs):
        ax.text(
            1.5, y_coords[idx], inp, ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f1f5f9", edgecolor="#64748b", lw=1.5)
        )
        
    # Explainability Agent in the center (draw a large box containing items)
    # We will use ax.text to label the agent box, and draw component boxes inside
    ax.text(
        5.5, 2.7, "Explainability Agent\n\n\n\n\n\n", ha='center', va='center', fontsize=12, fontweight='bold', color="#0e7490",
        bbox=dict(boxstyle="round,pad=2.2", facecolor="#ecfeff", edgecolor="#0891b2", lw=2.0)
    )
    
    # Internal explainability components
    components = ["1. Grad-CAM (Visual)", "2. Swin Attention Grid", "3. SHAP attribution"]
    comp_y = [3.2, 2.5, 1.8]
    for idx, comp in enumerate(components):
        ax.text(
            5.5, comp_y[idx], comp, ha='center', va='center', fontsize=9, fontweight='semibold',
            bbox=dict(boxstyle="square,pad=0.3", facecolor="#ffffff", edgecolor="#0891b2", lw=1.0)
        )
        
    # Output package on the right
    ax.text(
        9.5, 2.7, "Explainability Package\n(Xe)", ha='center', va='center', fontsize=10, fontweight='bold', color="#0f766e",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0fdfa", edgecolor="#0d9488", lw=1.5)
    )
    
    # Draw connections (lines with arrows)
    for idx in range(len(inputs)):
        ax.annotate("", xy=(3.8, 2.7), xytext=(2.7, y_coords[idx]), arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.2))
        
    ax.annotate("", xy=(8.5, 2.7), xytext=(7.2, 2.7), arrowprops=dict(arrowstyle="-|>", color="#0891b2", lw=1.5))
    
    ax.set_xlim(0, 11)
    ax.set_ylim(0.2, 5.5)
    
    plt.tight_layout()
    plt.savefig("04_Figures/figure7_explainability.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Figure 7: Explainability agent schematic.")

def generate_tables():
    """Generates formatted Tables in 05_Tables"""
    os.makedirs("05_Tables", exist_ok=True)
    
    # Table 1: Severity classification
    sev_data = {
        "Severity Score Range": ["0.00 - 0.20", "0.21 - 0.40", "0.41 - 0.60", "0.61 - 0.80", "0.81 - 1.00"],
        "Category": ["Healthy", "Mild", "Moderate", "Severe", "Critical"]
    }
    df_sev = pd.DataFrame(sev_data)
    df_sev.to_csv("05_Tables/severity_categories.csv", index=False)
    
    # LaTeX format Table 1
    latex_sev = df_sev.to_latex(index=False, caption="Crop Damage Severity Categories", label="tab:severity")
    with open("05_Tables/severity_categories.tex", "w") as f:
        f.write(latex_sev)
        
    # Table 2: Weight profiles
    weight_data = {
        "Crop Type": ["Rice", "Cotton", "Wheat", "Default (Other)"],
        "Visual Weight (alpha)": [0.30, 0.50, 0.30, 0.40],
        "Context Weight (beta)": [0.40, 0.20, 0.20, 0.30],
        "Severity Weight (gamma)": [0.30, 0.30, 0.50, 0.30]
    }
    df_weight = pd.DataFrame(weight_data)
    df_weight.to_csv("05_Tables/crop_weight_profiles.csv", index=False)
    
    # LaTeX format Table 2
    latex_weight = df_weight.to_latex(index=False, caption="Adaptive Fusion Weight Profiles by Crop Type", label="tab:weights")
    with open("05_Tables/crop_weight_profiles.tex", "w") as f:
        f.write(latex_weight)
        
    print("Generated Tables (CSV & LaTeX) inside 05_Tables.")

if __name__ == "__main__":
    create_workflow_diagram()
    create_explainability_diagram()
    generate_tables()
