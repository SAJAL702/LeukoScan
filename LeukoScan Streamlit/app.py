import streamlit as st
import os
from PIL import Image
from phase3_pipeline import run_phase3

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="LeukoScan AI", page_icon="🔬", layout="centered")

st.title("🔬 LeukoScan: ALL Diagnosis System")
st.markdown("**Phase 3 Integration:** U-Net Segmentation (Phase 2) & ResNet-50 Classification (Phase 1) with Grad-CAM")

# Ensure folders exist for OpenCV to read/write
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- 2. FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload Peripheral Blood Smear Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily so our Phase 3 script can read it via cv2
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Show the original image
    st.image(uploaded_file, caption="Original Uploaded Image", use_container_width=True)
    
    # --- 3. RUN AI PIPELINE ---
    # st.spinner creates a beautiful loading animation while the AI works
    with st.spinner('AI Engine is analyzing cells... Please wait.'):
        report = run_phase3(file_path)
        
    # --- 4. DISPLAY RESULTS ---
    if report and "Error" not in report["diagnosis"]:
        st.success("Analysis Complete!")
        
        st.markdown("### 📊 Diagnostic Report")
        
        # Create 3 neat columns for the metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("WHO Diagnosis", report["diagnosis"])
        col2.metric("Blast Cells Found", report["blast_count"])
        
        
        st.markdown("### 👁️ AI Explainability Map")
        st.markdown("*Red bounding boxes indicate Blasts. Heatmaps highlight the nuclear chromatin texture used for classification.*")
        
        # Display the final image with boxes and Grad-CAM
        st.image(report["result_image"], use_container_width=True)
        
    else:
        error_msg = report.get('diagnosis', 'Unknown Error') if report else "Pipeline failed."
        st.error(f"Something went wrong! Error: {error_msg}")