import streamlit as st
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile

# --- 1. CONFIGURATION & STATE MANAGEMENT ---
st.set_page_config(layout="wide", page_title="Advanced Image Processor", page_icon="🎨")

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main .block-container {
        padding-top: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        margin-top: 1rem;
    }
    .step-card {
        background: white;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .step-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'compare_mode' not in st.session_state:
    st.session_state['compare_mode'] = False
if 'compare_indices' not in st.session_state:
    st.session_state['compare_indices'] = [0, -1]

def reset_app():
    st.session_state['history'] = []
    st.session_state['compare_mode'] = False
    st.rerun()

def undo_last():
    if len(st.session_state['history']) > 1:
        st.session_state['history'].pop()
        st.rerun()

# --- 2. ENHANCED NOISE FUNCTIONS ---
def add_noise(image, noise_type, intensity=0.1):
    """Enhanced noise functions with better implementations"""
    row, col, ch = image.shape
    img_float = image.astype(np.float32) / 255.0
    noisy_img = img_float.copy()

    if noise_type == "Gaussian":
        mean = 0
        sigma = intensity
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy_img = img_float + gauss

    elif noise_type == "Impulse":
        prob = intensity
        mask = np.random.rand(row, col) < prob
        noise = np.random.rand(row, col, ch)
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        noisy_img[mask_3d] = noise[mask_3d]

    elif noise_type == "Uniform":
        uni = np.random.uniform(-intensity, intensity, (row, col, ch))
        noisy_img = img_float + uni

    elif noise_type == "Rayleigh":
        scale = intensity
        rayleigh = np.random.rayleigh(scale, (row, col, ch))
        noisy_img = img_float + rayleigh

    elif noise_type == "Gamma":
        shape = 2.0
        scale = intensity
        gamma_noise = np.random.gamma(shape, scale, (row, col, ch))
        noisy_img = img_float + gamma_noise

    elif noise_type == "Exponential":
        scale = intensity
        exp_noise = np.random.exponential(scale, (row, col, ch))
        noisy_img = img_float + exp_noise

    elif noise_type == "Salt-and-Pepper":
        s_vs_p = 0.5
        amount = intensity
        out = image.copy()
        
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 255

        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0
        return out

    elif noise_type == "Poisson":
        vals = len(np.unique(img_float))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy_img = np.random.poisson(img_float * vals) / float(vals)

    elif noise_type == "Speckle":
        gauss = np.random.randn(row, col, ch)
        noisy_img = img_float + img_float * gauss * intensity

    noisy_img = np.clip(noisy_img, 0.0, 1.0)
    return (noisy_img * 255).astype(np.uint8)

# --- 3. ENHANCED FILTER FUNCTIONS ---
def apply_filter(image, filter_type, kernel_size=3, **kwargs):
    """Enhanced filtering with more options"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    if filter_type == "Gaussian Filter":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif filter_type == "Median Filter":
        return cv2.medianBlur(image, kernel_size)
    
    elif filter_type == "Max Filter":
        return maximum_filter(image, size=kernel_size)
    
    elif filter_type == "Min Filter":
        return minimum_filter(image, size=kernel_size)
    
    elif filter_type == "Bilateral Filter":
        d = kernel_size
        sigmaColor = kwargs.get('sigmaColor', 75)
        sigmaSpace = kwargs.get('sigmaSpace', 75)
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    
    elif filter_type == "Laplacian":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
    
    elif filter_type == "Sobel X":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobelx = np.uint8(np.absolute(sobelx))
        return cv2.cvtColor(sobelx, cv2.COLOR_GRAY2RGB)
    
    elif filter_type == "Sobel Y":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        sobely = np.uint8(np.absolute(sobely))
        return cv2.cvtColor(sobely, cv2.COLOR_GRAY2RGB)
    
    elif filter_type == "Canny Edge":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    elif filter_type == "Sharpen":
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    elif filter_type == "Box Filter":
        return cv2.boxFilter(image, -1, (kernel_size, kernel_size))
    
    return image

# --- 4. IMAGE TRANSFORMATIONS ---
def apply_transformation(image, transform_type, **kwargs):
    """Apply geometric and color transformations"""
    if transform_type == "Rotate 90°":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    elif transform_type == "Rotate 180°":
        return cv2.rotate(image, cv2.ROTATE_180)
    
    elif transform_type == "Rotate 270°":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    elif transform_type == "Flip Horizontal":
        return cv2.flip(image, 1)
    
    elif transform_type == "Flip Vertical":
        return cv2.flip(image, 0)
    
    elif transform_type == "Grayscale":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    elif transform_type == "Invert":
        return 255 - image
    
    elif transform_type == "Brightness":
        value = kwargs.get('value', 1.0)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    elif transform_type == "Contrast":
        value = kwargs.get('value', 1.0)
        return np.clip(image.astype(np.float32) * value, 0, 255).astype(np.uint8)
    
    elif transform_type == "Histogram Equalization":
        img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    elif transform_type == "Sepia":
        kernel = np.array([[0.393, 0.769, 0.189],
                          [0.349, 0.686, 0.168],
                          [0.272, 0.534, 0.131]])
        sepia = cv2.transform(image, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    
    return image

# --- 5. MORPHOLOGICAL OPERATIONS ---
def apply_morphology(image, morph_type, kernel_size=3):
    """Apply morphological operations"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if morph_type == "Erosion":
        result = cv2.erode(gray, kernel, iterations=1)
    elif morph_type == "Dilation":
        result = cv2.dilate(gray, kernel, iterations=1)
    elif morph_type == "Opening":
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif morph_type == "Closing":
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    elif morph_type == "Gradient":
        result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    elif morph_type == "Top Hat":
        result = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    elif morph_type == "Black Hat":
        result = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    else:
        result = gray
    
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

# --- 6. ANALYSIS FUNCTIONS ---
def calculate_metrics(original, processed):
    """Calculate image quality metrics"""
    # Ensure both images have the same dimensions
    if original.shape != processed.shape:
        # Resize processed to match original
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    # Convert to grayscale for calculations
    orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float64)
    proc_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY).astype(np.float64)
    
    # MSE
    mse = np.mean((orig_gray - proc_gray) ** 2)
    
    # PSNR
    if mse == 0:
        psnr = 100
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    # MAE
    mae = np.mean(np.abs(orig_gray - proc_gray))
    
    return {"MSE": mse, "PSNR": psnr, "MAE": mae}

def show_histogram(image, title="Histogram"):
    """Generate histogram visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    colors = ('red', 'green', 'blue')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, alpha=0.7, linewidth=1.5)
    ax.set_xlim([0, 256])
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

# --- 7. EXPORT FUNCTIONS ---
def export_all_images():
    """Export all images as a zip file"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for idx, (label, img_arr) in enumerate(st.session_state['history']):
            img_pil = Image.fromarray(img_arr)
            img_buffer = BytesIO()
            img_pil.save(img_buffer, format='PNG')
            zip_file.writestr(f"step_{idx}_{label.replace(' ', '_').replace('+', '')}.png", 
                            img_buffer.getvalue())
    return zip_buffer.getvalue()

# --- 8. FRONTEND UI ---

st.title("🎨 Advanced Digital Image Processing Lab")
st.markdown("**Professional image processing tool for DIP course** - Apply noise, filters, transformations, and morphological operations")

# Sidebar Controls
with st.sidebar:
    st.header("🖼️ Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg', 'bmp', 'tiff'])

    if uploaded_file is not None:
        if not st.session_state['history']:
            image = Image.open(uploaded_file).convert('RGB')
            img_array = np.array(image)
            st.session_state['history'].append(("Original", img_array))

    st.divider()

    if st.session_state['history']:
        st.header("⚙️ Operations")
        
        # Operation Category
        op_category = st.selectbox("Category", 
            ["🔊 Add Noise", "🔧 Apply Filter", "🎭 Transform", "🔬 Morphology"])
        
        if op_category == "🔊 Add Noise":
            noise_opts = ["Gaussian", "Impulse", "Uniform", "Rayleigh", "Gamma", 
                         "Exponential", "Salt-and-Pepper", "Poisson", "Speckle"]
            selected_noise = st.selectbox("Noise Type", noise_opts)
            intensity = st.slider("Intensity", 0.01, 0.5, 0.1, 0.01)
            
            if st.button(f"➕ Apply {selected_noise}", use_container_width=True):
                last_label, last_img = st.session_state['history'][-1]
                new_img = add_noise(last_img, selected_noise, intensity)
                st.session_state['history'].append((f"{selected_noise} Noise", new_img))
                st.rerun()

        elif op_category == "🔧 Apply Filter":
            filter_opts = ["Gaussian Filter", "Median Filter", "Max Filter", "Min Filter",
                          "Bilateral Filter", "Laplacian", "Sobel X", "Sobel Y", 
                          "Canny Edge", "Sharpen", "Box Filter"]
            selected_filter = st.selectbox("Filter Type", filter_opts)
            k_size = st.slider("Kernel Size", 3, 15, 5, step=2)
            
            # Additional parameters for bilateral filter
            extra_params = {}
            if selected_filter == "Bilateral Filter":
                extra_params['sigmaColor'] = st.slider("Sigma Color", 10, 200, 75)
                extra_params['sigmaSpace'] = st.slider("Sigma Space", 10, 200, 75)
            
            if st.button(f"➕ Apply {selected_filter}", use_container_width=True):
                last_label, last_img = st.session_state['history'][-1]
                new_img = apply_filter(last_img, selected_filter, k_size, **extra_params)
                st.session_state['history'].append((selected_filter, new_img))
                st.rerun()

        elif op_category == "🎭 Transform":
            transform_opts = ["Rotate 90°", "Rotate 180°", "Rotate 270°", 
                            "Flip Horizontal", "Flip Vertical", "Grayscale", 
                            "Invert", "Brightness", "Contrast", 
                            "Histogram Equalization", "Sepia"]
            selected_transform = st.selectbox("Transform Type", transform_opts)
            
            extra_params = {}
            if selected_transform == "Brightness":
                extra_params['value'] = st.slider("Brightness Factor", 0.1, 3.0, 1.5, 0.1)
            elif selected_transform == "Contrast":
                extra_params['value'] = st.slider("Contrast Factor", 0.1, 3.0, 1.5, 0.1)
            
            if st.button(f"➕ Apply {selected_transform}", use_container_width=True):
                last_label, last_img = st.session_state['history'][-1]
                new_img = apply_transformation(last_img, selected_transform, **extra_params)
                st.session_state['history'].append((selected_transform, new_img))
                st.rerun()

        elif op_category == "🔬 Morphology":
            morph_opts = ["Erosion", "Dilation", "Opening", "Closing", 
                         "Gradient", "Top Hat", "Black Hat"]
            selected_morph = st.selectbox("Morphology Type", morph_opts)
            k_size = st.slider("Kernel Size", 3, 15, 5, step=2)
            
            if st.button(f"➕ Apply {selected_morph}", use_container_width=True):
                last_label, last_img = st.session_state['history'][-1]
                new_img = apply_morphology(last_img, selected_morph, k_size)
                st.session_state['history'].append((selected_morph, new_img))
                st.rerun()

        st.divider()
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↩️ Undo", use_container_width=True, disabled=len(st.session_state['history']) <= 1):
                undo_last()
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                reset_app()

        st.divider()
        
        # Analysis & Export
        st.header("📊 Analysis & Export")
        
        if st.checkbox("📈 Compare Images"):
            st.session_state['compare_mode'] = True
            if len(st.session_state['history']) >= 2:
                idx1 = st.selectbox("First Image", range(len(st.session_state['history'])), 
                                   format_func=lambda x: f"Step {x}: {st.session_state['history'][x][0]}")
                idx2 = st.selectbox("Second Image", range(len(st.session_state['history'])), 
                                   index=len(st.session_state['history'])-1,
                                   format_func=lambda x: f"Step {x}: {st.session_state['history'][x][0]}")
                st.session_state['compare_indices'] = [idx1, idx2]
        else:
            st.session_state['compare_mode'] = False
        
        if len(st.session_state['history']) > 0:
            st.download_button(
                label="💾 Download All Images",
                data=export_all_images(),
                file_name="image_processing_results.zip",
                mime="application/zip",
                use_container_width=True
            )

# --- 9. MAIN DISPLAY AREA ---

if not st.session_state['history']:
    st.info("👈 **Please upload an image in the sidebar to get started!**")
    st.markdown("""
    ### Features:
    - 🔊 **9 Noise Types**: Gaussian, Impulse, Uniform, Rayleigh, Gamma, Exponential, Salt-and-Pepper, Poisson, Speckle
    - 🔧 **11 Filters**: Gaussian, Median, Max, Min, Bilateral, Laplacian, Sobel, Canny, Sharpen, Box
    - 🎭 **11 Transformations**: Rotations, Flips, Color adjustments, Histogram Equalization
    - 🔬 **7 Morphological Operations**: Erosion, Dilation, Opening, Closing, Gradient, Top/Black Hat
    - 📊 **Analysis Tools**: Histograms, Image Quality Metrics (MSE, PSNR, MAE)
    - 💾 **Export**: Download individual images or all at once as ZIP
    """)
else:
    # Compare Mode
    if st.session_state['compare_mode'] and len(st.session_state['history']) >= 2:
        st.write("## 🔍 Image Comparison")
        
        idx1, idx2 = st.session_state['compare_indices']
        label1, img1 = st.session_state['history'][idx1]
        label2, img2 = st.session_state['history'][idx2]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### Step {idx1}: {label1}")
            st.image(img1, use_container_width=True, channels="RGB")
            with st.expander("📊 Show Histogram"):
                fig1 = show_histogram(img1, f"Histogram - {label1}")
                st.pyplot(fig1)
        
        with col2:
            st.markdown(f"### Step {idx2}: {label2}")
            st.image(img2, use_container_width=True, channels="RGB")
            with st.expander("📊 Show Histogram"):
                fig2 = show_histogram(img2, f"Histogram - {label2}")
                st.pyplot(fig2)
        
        # Metrics comparison
        if idx1 == 0 or idx2 == 0:
            metrics = calculate_metrics(img1 if idx1 == 0 else img2, 
                                       img2 if idx1 == 0 else img1)
            st.write("### 📈 Quality Metrics")
            met_col1, met_col2, met_col3 = st.columns(3)
            with met_col1:
                st.metric("MSE", f"{metrics['MSE']:.2f}")
            with met_col2:
                st.metric("PSNR", f"{metrics['PSNR']:.2f} dB")
            with met_col3:
                st.metric("MAE", f"{metrics['MAE']:.2f}")
        
        st.divider()
    
    # Processing Chain Display
    history_len = len(st.session_state['history'])
    st.write(f"## 🎬 Processing Chain ({history_len} steps)")
    
    # Create scrollable columns
    cols = st.columns(min(history_len, 5))
    
    for idx, (label, img_arr) in enumerate(st.session_state['history']):
        with cols[idx % 5]:
            is_latest = idx == history_len - 1
            border_style = "border: 3px solid #667eea;" if is_latest else ""
            
            st.markdown(f"""
            <div style="{border_style} padding: 10px; border-radius: 8px; background: white;">
                <h4 style="margin: 0;">Step {idx}</h4>
                <p style="color: #667eea; margin: 0;"><b>{label}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.image(img_arr, use_container_width=True, channels="RGB")
            
            # Download individual image
            img_pil = Image.fromarray(img_arr)
            img_buffer = BytesIO()
            img_pil.save(img_buffer, format='PNG')
            st.download_button(
                label="⬇️",
                data=img_buffer.getvalue(),
                file_name=f"step_{idx}_{label.replace(' ', '_')}.png",
                mime="image/png",
                key=f"download_{idx}",
                use_container_width=True
            )
            
            # Show histogram in expander
            with st.expander("📊 Histogram"):
                fig = show_histogram(img_arr, f"{label}")
                st.pyplot(fig)
                plt.close(fig)
        
        # Start new row after 5 columns
        if (idx + 1) % 5 == 0 and idx + 1 < history_len:
            cols = st.columns(min(history_len - (idx + 1), 5))

    # Final metrics if we have both original and processed
    if history_len > 1:
        st.divider()
        st.write("### 📊 Final Quality Metrics (Original vs Latest)")
        original_img = st.session_state['history'][0][1]
        final_img = st.session_state['history'][-1][1]
        metrics = calculate_metrics(original_img, final_img)
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Mean Squared Error", f"{metrics['MSE']:.4f}", 
                     help="Lower is better - measures average squared difference")
        with metric_col2:
            st.metric("Peak SNR", f"{metrics['PSNR']:.2f} dB", 
                     help="Higher is better - measures signal to noise ratio")
        with metric_col3:
            st.metric("Mean Absolute Error", f"{metrics['MAE']:.4f}", 
                     help="Lower is better - measures average absolute difference")