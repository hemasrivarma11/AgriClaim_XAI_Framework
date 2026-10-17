import os
import sys
from datetime import datetime
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Adjust path to import packages from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from agents.data_agent import DataIntegrationAgent
from agents.vision_agent import CropVisionAgent
from agents.context_agent import ContextVerificationAgent
from agents.severity_agent import SeverityAgent
from agents.insurance_agent import ClaimReasoningAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.report_agent import ReportAgent
from utils.metrics_logger import MetricsLogger
import time

# Custom Page Configuration
st.set_page_config(
    page_title="AgriClaim-XAI Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #38bdf8;
        background-color: #334155;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        background-color: #0f172a !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #38bdf8;
    }
    .metric-val {
        font-size: 24px;
        font-weight: bold;
        color: #f8fafc;
    }
    .metric-title {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .decision-approve {
        background-color: #064e3b;
        border: 1px solid #059669;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        color: #34d399;
        font-weight: bold;
        font-size: 24px;
    }
    .decision-review {
        background-color: #7c2d12;
        border: 1px solid #d97706;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        color: #fbbf24;
        font-weight: bold;
        font-size: 24px;
    }
    .decision-reject {
        background-color: #7f1d1d;
        border: 1px solid #dc2626;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        color: #f87171;
        font-weight: bold;
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("🌱 AgriClaim-XAI: Multi-Agent Insurance Decision Support")
st.markdown("An Explainable Multi-Agent Framework for automated crop damage validation and insurance claims reasoning.")
st.markdown("---")

# Sidebar - User Inputs
st.sidebar.header("📋 Input Configuration")

# Image Selection Mode
image_source = st.sidebar.radio("Select Crop Image Source:", ["Samples Folder", "Upload Custom Image"])

# Prepare sample list
samples_dir = "samples"
sample_files = []
if os.path.exists(samples_dir):
    sample_files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

selected_image_path = None
uploaded_file = None

if image_source == "Samples Folder" and sample_files:
    selected_sample = st.sidebar.selectbox("Choose a Sample Image:", sample_files)
    selected_image_path = os.path.join(samples_dir, selected_sample)
    
    # Pre-select crop type based on sample name for user ease
    if "rice" in selected_sample.lower():
        default_crop = "Rice"
        default_dmg = "Leaf Blight"
    elif "cotton" in selected_sample.lower():
        default_crop = "Cotton"
        default_dmg = "Pest Infestation"
    elif "wheat" in selected_sample.lower():
        default_crop = "Wheat"
        default_dmg = "Leaf Rust"
    else:
        default_crop = "Rice"
        default_dmg = "Healthy"
else:
    uploaded_file = st.sidebar.file_uploader("Upload Crop Image:", type=["png", "jpg", "jpeg"])
    default_crop = "Rice"
    default_dmg = "Leaf Blight"

# Metadata inputs
PRESET_CROPS = ["Rice", "Cotton", "Wheat", "Maize", "Soybean", "Sugarcane", "Custom..."]

# Determine default index; fall back to 0 (Rice) if not in list
default_crop_index = PRESET_CROPS.index(default_crop) if default_crop in PRESET_CROPS else 0
crop_choice = st.sidebar.selectbox("Crop Type:", PRESET_CROPS, index=default_crop_index)

if crop_choice == "Custom...":
    crop_type = st.sidebar.text_input(
        "Enter your crop name:",
        placeholder="e.g. Turmeric, Banana, Groundnut...",
        help="Type any crop name. The framework will apply the default adaptive fusion weights."
    ).strip()
    if not crop_type:
        st.sidebar.warning("Please enter a crop name to continue.")
        crop_type = "Unknown"
else:
    crop_type = crop_choice

season = st.sidebar.selectbox("Season:", ["Monsoon", "Winter", "Summer"])
growth_stage = st.sidebar.selectbox("Crop Growth Stage:", ["Seedling", "Vegetative", "Flowering", "Fruiting", "Harvest"])

st.sidebar.subheader("📍 Location Settings")
col_lat, col_lon = st.sidebar.columns(2)
with col_lat:
    lat = st.number_input("Latitude:", min_value=8.0, max_value=38.0, value=17.3850, step=0.01)
with col_lon:
    lon = st.number_input("Longitude:", min_value=68.0, max_value=97.0, value=78.4867, step=0.01)

st.sidebar.subheader("🌦️ Environmental Simulation Context")
suspected_damage = st.sidebar.selectbox(
    "Suspected/Reported Damage Type:",
    ["Healthy", "Leaf Blight", "Pest Infestation", "Leaf Rust", "Flood Damage", "Drought Damage"],
    index=["Healthy", "Leaf Blight", "Pest Infestation", "Leaf Rust", "Flood Damage", "Drought Damage"].index(default_dmg)
)

# Custom weights tuning (optional expander)
with st.sidebar.expander("⚙️ Manual Evidence Weights Override (Optional)"):
    st.info("By default, the framework uses crop-specific adaptive fusion weights.")
    override_weights = st.checkbox("Override Adaptive Weights")
    w_vis = st.slider("Visual Weight (α):", 0.0, 1.0, 0.4, 0.05)
    w_ctx = st.slider("Context Weight (β):", 0.0, 1.0, 0.3, 0.05)
    w_sev = st.slider("Severity Weight (γ):", 0.0, 1.0, 0.3, 0.05)
    
    # Weight normalization warning
    tot_w = w_vis + w_ctx + w_sev
    if abs(tot_w - 1.0) > 1e-4:
        st.warning(f"Weights sum up to {tot_w:.2f}. They will be normalized to 1.0 during reasoning.")

# Run button
run_pipeline = st.sidebar.button("🚀 Execute Framework Pipeline", type="primary")

# Execute Pipeline logic
if run_pipeline or 'pipeline_run' in st.session_state:
    st.session_state['pipeline_run'] = True
    
    # Resolve image path
    image_temp_path = "temp_claim_image.png"
    if image_source == "Upload Custom Image" and uploaded_file is not None:
        # Save uploaded file temporarily
        with open(image_temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        img_path_to_run = image_temp_path
    elif selected_image_path:
        img_path_to_run = selected_image_path
    else:
        st.error("Please select or upload a crop image.")
        st.stop()

    # Step-by-step progress simulation (visually engaging)
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Instantiation
    status_text.text("1. Instantiating Agents...")
    progress_bar.progress(10)
    data_agent = DataIntegrationAgent()
    vision_agent = CropVisionAgent()
    context_agent = ContextVerificationAgent()
    severity_agent = SeverityAgent()
    claim_agent = ClaimReasoningAgent()
    xai_agent = ExplainabilityAgent()
    report_agent = ReportAgent()

    t_start = time.time()

    # Step 1
    status_text.text("2. Running Data Integration Agent...")
    progress_bar.progress(25)
    gps_coords = {"latitude": lat, "longitude": lon}
    metadata_dict = {
        "crop_type": crop_type,
        "season": season,
        "growth_stage": growth_stage,
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    processed_dataset = data_agent.process(img_path_to_run, gps_coords, metadata_dict, suspected_damage)

    # Step 2
    status_text.text("3. Analyzing crop leaf with Swin Transformer (Vision Agent)...")
    progress_bar.progress(40)
    visual_evidence = vision_agent.analyze(processed_dataset, img_path_to_run)

    # Step 3
    status_text.text("4. Performing Context & Weather Verification...")
    progress_bar.progress(55)
    context_evidence = context_agent.verify(visual_evidence, processed_dataset["weather"], processed_dataset["metadata"], suspected_damage=suspected_damage)

    # Step 4
    status_text.text("5. Estimating Severity & Expected Yield Loss...")
    progress_bar.progress(70)
    severity_evidence = severity_agent.estimate(visual_evidence, context_evidence, processed_dataset["metadata"])

    # Step 5
    status_text.text("6. Running Evidence Fusion and Claim Reasoning...")
    progress_bar.progress(85)
    # Manual weights override check
    if override_weights:
        # Override weights inside visual/context/severity evidence structure or reasoning logic
        # We can pass custom weights to reasoning engine via a mock override
        norm_factor = w_vis + w_ctx + w_sev
        claim_agent.evidence_fusion_engine.crop_weight_profiles[crop_type.lower()] = {
            "alpha": w_vis / norm_factor,
            "beta": w_ctx / norm_factor,
            "gamma": w_sev / norm_factor
        }
    claim_decision = claim_agent.process_claim(visual_evidence, context_evidence, severity_evidence, processed_dataset["metadata"])

    # Step 6
    status_text.text("7. Generating Grad-CAM and SHAP Decision Explanations...")
    progress_bar.progress(95)
    explainability_pkg = xai_agent.explain(
        raw_image=processed_dataset["raw_image"],
        visual_evidence=visual_evidence,
        context_evidence=context_evidence,
        severity_evidence=severity_evidence,
        claim_decision=claim_decision,
        metadata=processed_dataset["metadata"]
    )

    # Step 7
    status_text.text("8. Generating Reports...")
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()

    report_pkg = report_agent.generate(
        claim_decision=claim_decision,
        visual_evidence=visual_evidence,
        context_evidence=context_evidence,
        severity_evidence=severity_evidence,
        explainability_pkg=explainability_pkg,
        metadata=processed_dataset["metadata"]
    )

    # Save to 08_Results
    results_dir = "08_Results"
    os.makedirs(results_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path_to_run))[0]
    json_report_path = os.path.join(results_dir, f"{base_name}_claim_report.json")
    html_report_path = os.path.join(results_dir, f"{base_name}_claim_report.html")
    report_agent.export_json(report_pkg, json_report_path)
    report_agent.export_html(report_pkg, html_report_path)

    # Log metrics
    exec_time = time.time() - t_start
    logger = MetricsLogger()
    logger.log_prediction(
        image_id=base_name,
        crop_type=crop_type,
        visual_evidence=visual_evidence.get("disease", "Healthy"),
        confidence_score=visual_evidence.get("confidence", 0.0),
        severity=severity_evidence.get("Severity", 0.0),
        yield_loss=severity_evidence.get("YieldLoss", 0.0),
        cci=context_evidence.get("cci", 0.0),
        claim_reasoning_score=claim_decision.get("CRS", 0.0),
        decision=claim_decision.get("Decision", "Reject"),
        claim_category=claim_decision.get("ClaimCategory", "No Compensation"),
        execution_time=exec_time
    )

    # Display Dashboard Panels
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📸 Crop Input & Details")
        st.image(processed_dataset["resized_image"], caption="Resized input (224x224)", use_column_width=True)
        
        # Decision Banner
        decision = claim_decision["Decision"]
        crs = claim_decision["CRS"]
        
        if "Invalid Image" in claim_decision.get("ClaimCategory", ""):
            st.markdown(f'<div class="decision-reject">CLAIM REJECTED<br><span style="font-size:16px;">INVALID INPUT: Non-Crop Image Uploaded</span></div>', unsafe_allow_html=True)
        elif decision == "Approve":
            st.markdown(f'<div class="decision-approve">CLAIM APPROVED<br><span style="font-size:16px;">CRS: {crs:.4f} ({claim_decision["ClaimCategory"]})</span></div>', unsafe_allow_html=True)
        elif decision == "Manual Review":
            st.markdown(f'<div class="decision-review">MANUAL REVIEW REQUIRED<br><span style="font-size:16px;">CRS: {crs:.4f} (Pending Verification)</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="decision-reject">CLAIM REJECTED<br><span style="font-size:16px;">CRS: {crs:.4f} (No Compensation)</span></div>', unsafe_allow_html=True)
        
        # Quick summary stats
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Detected Cause:** {visual_evidence['disease']}")
        st.markdown(f"**Vulnerability Factor:** {severity_evidence['vulnerability_multiplier']:.2f}")

    with col2:
        # Tabbed workspace for Agents
        tab_data, tab_vision, tab_context, tab_severity, tab_fusion, tab_xai, tab_reports = st.tabs([
            "📁 Data Sync", "👁️ Crop Vision", "🌦️ Context Verification", "📉 Yield Impact", "⚖️ Claim Fusion", "🧠 Explainability", "📄 Reports"
        ])

        with tab_data:
            st.subheader("Data Integration Agent (Agent 1)")
            st.markdown("Synchronizes image pixels, weather datasets, and metadata into a synchronized representation.")
            
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                st.markdown("**Validated Metadata**")
                for k, v in processed_dataset["metadata"].items():
                    st.write(f"- **{k.capitalize().replace('_', ' ')}:** {v}")
            
            with c_d2:
                st.markdown("**Retrieved Weather Observations**")
                for k, v in processed_dataset["weather"].items():
                    if k != "station_id":
                        st.write(f"- **{k.capitalize().replace('_', ' ')}:** {v}")
                st.write(f"Ref Station: `{processed_dataset['weather']['station_id']}`")
            
            st.markdown("---")
            st.markdown("**Swin Patch Embedding Simulation**")
            patches = processed_dataset["image_patches"]
            st.write(f"Divided the 224x224 image into non-overlapping patches. Displaying first 16 patches:")
            cols = st.columns(8)
            for idx in range(min(16, len(patches))):
                with cols[idx % 8]:
                    st.image(patches[idx], clamp=True, use_column_width=True)

        with tab_vision:
            st.subheader("Crop Vision Agent (Agent 2)")
            st.markdown("Leverages a **Swin Transformer** backbone to analyze crop images for signs of visual damage.")
            
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Identified Disease</div>
                    <div class="metric-val">{visual_evidence['disease']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Model Confidence</div>
                    <div class="metric-val">{visual_evidence['confidence']*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_v2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Damage Severity Score</div>
                    <div class="metric-val">{visual_evidence['severity']*100:.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Feature Extraction Model</div>
                    <div class="metric-val">{visual_evidence['swin_backbone']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("**Swin Transformer Linear Classifier Layer:**")
            st.code("Loss = L_cls + λ * L_sev | Output = {DiseaseCategory, SeverityScore, Confidence}")

        with tab_context:
            st.subheader("Context Verification Agent (Agent 3)")
            st.markdown("Validates visual findings against weather conditions, coordinates, and seasonal records.")

            # ── Impossibility Alert (shown before CCI) ──────────────────────────
            if context_evidence.get("impossibility_flag"):
                st.error(
                    f"🚫 **CLIMATOLOGICAL IMPOSSIBILITY DETECTED — Automatic Rejection**\n\n"
                    f"{context_evidence['impossibility_reason']}\n\n"
                    f"**CCI forced to 0.05 → CRS will fall below 0.50 → Claim: NO COMPENSATION.**"
                )
            
            # CCI Gauge representation
            cci_val = context_evidence["cci"]
            st.metric("Context Consistency Index (CCI / CVI)", f"{cci_val:.4f}", 
                      delta="Consistent Context" if context_evidence["verified_context"] else "Inconsistent Context",
                      delta_color="normal" if context_evidence["verified_context"] else "inverse")
            
            st.markdown("**Consistency Component Breakdown:**")
            c_c1, c_c2 = st.columns(2)
            with c_c1:
                st.write(f"- Weather Score ($S_w$): **{context_evidence['scores']['weather_score']:.2f}**")
                st.write(f"- Location Score ($S_l$): **{context_evidence['scores']['location_score']:.2f}**")
                st.write(f"- Season Score ($S_s$): **{context_evidence['scores']['season_score']:.2f}**")
            with c_c2:
                st.write(f"- Metadata Score ($S_m$): **{context_evidence['scores']['metadata_score']:.2f}**")
                st.write(f"- Growth Stage Score ($S_{{gs}}$): **{context_evidence['scores'].get('growth_stage_score', 0.0):.2f}**")
            
            st.markdown("---")
            st.markdown("**Verification Logic Logs:**")
            for k, val in context_evidence["logs"].items():
                st.markdown(f"- **{k.capitalize().replace('_', ' ')}:** {val}")

        with tab_severity:
            st.subheader("Damage Severity & Yield Loss Agent (Agent 4)")
            st.markdown("Translates visual signs of damage and affected area into economically meaningful yield reduction indicators.")
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #f59e0b;">
                    <div class="metric-title">Categorized Severity</div>
                    <div class="metric-val">{severity_evidence['severity_category']}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s2:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #3b82f6;">
                    <div class="metric-title">Affected Area</div>
                    <div class="metric-val">{severity_evidence.get('AffectedArea', 0.0)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s3:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #ef4444;">
                    <div class="metric-title">Estimated Yield Loss</div>
                    <div class="metric-val">{severity_evidence['YieldLoss']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown(f"**Risk Level:** `{severity_evidence['RiskLevel']}`")
            st.markdown("---")
            st.markdown("**Vulnerability Model:**")
            st.latex(r"Y_L = S_d \times A_{area} \times \text{Vulnerability} + (1 - \text{CCI}) \times 0.10")

        with tab_fusion:
            st.subheader("Claim Reasoning Agent & Fusion Engine (Agent 5)")
            st.markdown("Integrates the multi-agent evidence vectors using adaptive weights to compute the Claim Reasoning Score.")
            
            st.metric("Claim Reasoning Score (CRS)", f"{crs:.4f}")
            
            # Show crop profile match note
            known_crops = ["rice", "cotton", "wheat", "maize", "soybean", "sugarcane"]
            if crop_type.lower() in known_crops:
                st.success(f"✅ Crop-specific adaptive weights loaded for **{crop_type}**.")
            else:
                st.info(f"ℹ️ Custom crop **'{crop_type}'** detected. Using **default** adaptive fusion weights (α=0.40, β=0.30, γ=0.30). You can fine-tune weights using the sidebar override panel.")
            
            # Display weights used
            st.markdown("**Adaptive Fusion Weights (Crop Dependent):**")
            evidence = claim_decision.get("Evidence", {})
            w_used = evidence.get("weights_used", {"alpha": 0.4, "beta": 0.3, "gamma": 0.3})
            
            col_w1, col_w2, col_w3 = st.columns(3)
            with col_w1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Visual Weight (α)</div>
                    <div class="metric-val">{w_used['alpha']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_w2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Context Weight (β)</div>
                    <div class="metric-val">{w_used['beta']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_w3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Severity Weight (γ)</div>
                    <div class="metric-val">{w_used['gamma']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("**Evidence Fusion Vector components:**")
            components = evidence.get("components", {})
            st.write(f"- Visual Component: **{components.get('visual_score', 0.0):.3f}**")
            st.write(f"- Context Component: **{components.get('context_score', 0.0):.3f}**")
            st.write(f"- Severity Component: **{components.get('severity_score', 0.0):.3f}**")
            st.latex(r"CRS = \alpha E_v + \beta E_c + \gamma E_s")
            
            st.markdown("**Claim Assessment Reason Codes (Task 4):**")
            reason_codes = claim_decision.get("ReasonCodes", [])
            if reason_codes:
                for code in reason_codes:
                    st.info(f"📌 {code}")
            else:
                st.write("- None")

        with tab_xai:
            st.subheader("Explainability Agent (Agent 6)")
            st.markdown("Provides visual, feature-level, and textual explanations for agricultural stakeholders and insurance adjusters.")
            
            st.markdown("### 1. Visual Interpretability Maps")
            col_x1, col_x2 = st.columns(2)
            with col_x1:
                st.image(explainability_pkg["gradcam_overlay"], caption="Grad-CAM Activation Map (Target Class)", use_column_width=True)
            with col_x2:
                st.image(explainability_pkg["attention_overlay"], caption="Swin Self-Attention Window Grid (7x7)", use_column_width=True)
                
            st.markdown("### 2. Decision Feature Attribution (SHAP Analysis)")
            shaps = explainability_pkg["shap_values"]
            
            # Plot SHAP contributions
            fig, ax = plt.subplots(figsize=(6, 2.2))
            features_names = ["Visual Score", "Context Index", "Severity Score"]
            values = [shaps["SHAP_Visual"], shaps["SHAP_Context"], shaps["SHAP_Severity"]]
            
            # Color-code based on positive/negative contributions
            colors = ['#10b981' if v >= 0 else '#ef4444' for v in values]
            y_pos = np.arange(len(features_names))
            
            ax.barh(y_pos, values, align='center', color=colors, height=0.4)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features_names, color='#f8fafc', fontsize=9)
            ax.axvline(0, color='#64748b', linewidth=0.8, linestyle='--')
            ax.set_xlabel('SHAP Feature Value Impact', color='#94a3b8', fontsize=8)
            ax.tick_params(colors='#94a3b8', labelsize=8)
            
            # Transparent background
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#334155')
            ax.spines['bottom'].set_color('#334155')
            
            st.pyplot(fig)
            st.write(f"**Base Decision Value (Neutral Case):** `{shaps['Base_Value']:.3f}` | **Current Claim Value:** `{shaps['Current_Value']:.3f}`")
            
            st.markdown("### 3. Decision Narrative Justification")
            st.write(explainability_pkg["text_explanation"])
            
            st.markdown("### 4. Top-5 Most Influential Factors (Task 5)")
            top_factors = explainability_pkg.get("top_influential_factors", [])
            for fac in top_factors:
                st.write(f"**{fac}**")

        with tab_reports:
            st.subheader("Report Generation Agent (Agent 7)")
            st.markdown("Compiles distinct outputs optimized for both farmers and insurance companies.")
            
            c_r1, c_r2 = st.columns(2)
            with c_r1:
                st.markdown("#### 👨‍🌾 Farmer Report Summary")
                st.json(report_pkg["farmer"])
            
            with c_r2:
                st.markdown("#### 🕵️ Insurance Officer Analytics")
                st.json(report_pkg["officer"])
                
            st.markdown("---")
            st.markdown("#### 📥 Export Formats Available")
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                with open(json_report_path, "r") as f:
                    st.download_button(
                        label="⬇️ Download Claim Report (JSON)",
                        data=f.read(),
                        file_name=f"{base_name}_claim_report.json",
                        mime="application/json",
                        use_container_width=True
                    )
            with col_b2:
                with open(html_report_path, "r", encoding='utf-8') as f:
                    st.download_button(
                        label="⬇️ Download Claim Report (HTML/PDF Ready)",
                        data=f.read(),
                        file_name=f"{base_name}_claim_report.html",
                        mime="text/html",
                        use_container_width=True
                    )
                    
            st.success(f"Reports saved in project output directory: `08_Results/`")
else:
    # Initial landing view
    st.info("👈 Select a crop leaf image, configure settings on the sidebar, and click **Execute Framework Pipeline** to run the multi-agent system!")
    
    # Display sample images preview if they exist
    if sample_files:
        st.markdown("### Preview of Available Sample Images:")
        cols_p = st.columns(4)
        for idx, sf in enumerate(sample_files):
            with cols_p[idx % 4]:
                st.image(os.path.join(samples_dir, sf), caption=sf.replace('_', ' ').replace('.png', '').title(), use_column_width=True)
