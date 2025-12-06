import streamlit as st
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

# --- 1. CONFIGURATION & STATE MANAGEMENT ---
st.set_page_config(layout="wide", page_title="Image Processor")

# Initialize session state to keep track of image history
if 'history' not in st.session_state:
    st.session_state['history'] = [] # List to store [ (label, image_array) ]

def reset_app():
    st.session_state['history'] = []
    st.rerun()

# --- 2. NOISE FUNCTIONS ---
def add_noise(image, noise_type, intensity=0.1):
    """
    Applies various noise types to an image.
    Image expected as valid uint8 numpy array (H, W, C).
    Intensity serves as a scaling factor for variance/std-dev.
    """
    row, col, ch = image.shape
    img_float = image.astype(np.float32) / 255.0  # Normalize to 0-1 for math
    
    noisy_img = img_float.copy()

    if noise_type == "Gaussian":
        mean = 0
        sigma = intensity # Use intensity as sigma
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy_img = img_float + gauss

    elif noise_type == "Impulse":
        # Random random values added to random pixels
        prob = intensity
        mask = np.random.rand(row, col) < prob
        noise = np.random.rand(row, col, ch)
        # Apply noise only where mask is True
        # We need to broadcast mask to 3 channels
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        noisy_img[mask_3d] = noise[mask_3d]

    elif noise_type == "Uniform":
        # Uniform distribution noise
        uni = np.random.uniform(-intensity, intensity, (row, col, ch))
        noisy_img = img_float + uni

    elif noise_type == "Rayleigh":
        # Rayleigh distribution (requires positive inputs usually)
        # We add it as additive noise
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
        # S&P affects specific pixels setting them to 0 or 1
        s_vs_p = 0.5
        amount = intensity
        out = image.copy()
        
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0
        return out # Return directly as uint8

    # Clip values to 0-1 range and convert back to uint8
    noisy_img = np.clip(noisy_img, 0.0, 1.0)
    return (noisy_img * 255).astype(np.uint8)

# --- 3. FILTER FUNCTIONS ---
def apply_filter(image, filter_type, kernel_size=3):
    """
    Applies filters to the image.
    Kernel size must be odd (3, 5, 7...).
    """
    if kernel_size % 2 == 0: kernel_size += 1 # Ensure odd
    
    if filter_type == "Gaussian Filter":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif filter_type == "Median Filter":
        return cv2.medianBlur(image, kernel_size)
    
    elif filter_type == "Max Filter":
        return maximum_filter(image, size=kernel_size)
    
    elif filter_type == "Min Filter":
        return minimum_filter(image, size=kernel_size)
    
    return image

# --- 4. FRONTEND UI ---

st.title("🖼️ Digital Image Processing Lab")
st.markdown("Upload an image, apply noise, filter it, and see the transformation history.")

# Sidebar Controls
with st.sidebar:
    st.header("1. Upload")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

    # Handle Upload
    if uploaded_file is not None:
        # Load image only if history is empty (first load)
        if not st.session_state['history']:
            image = Image.open(uploaded_file).convert('RGB')
            img_array = np.array(image)
            st.session_state['history'].append(("Original", img_array))

    st.divider()

    if st.session_state['history']:
        st.header("2. Operations")
        
        # Select Operation Type
        op_category = st.radio("Category", ["Add Noise", "Apply Filter"])
        
        if op_category == "Add Noise":
            noise_opts = ["Gaussian", "Impulse", "Uniform", "Rayleigh", "Gamma", "Exponential", "Salt-and-Pepper"]
            selected_noise = st.selectbox("Select Noise Type", noise_opts)
            intensity = st.slider("Noise Intensity", 0.01, 0.5, 0.05)
            
            if st.button(f"Apply {selected_noise}"):
                # Get last image
                last_label, last_img = st.session_state['history'][-1]
                # Apply
                new_img = add_noise(last_img, selected_noise, intensity)
                # Save to history
                st.session_state['history'].append((f"+ {selected_noise}", new_img))
                st.rerun()

        elif op_category == "Apply Filter":
            filter_opts = ["Gaussian Filter", "Median Filter", "Max Filter", "Min Filter"]
            selected_filter = st.selectbox("Select Filter", filter_opts)
            k_size = st.slider("Kernel Size", 3, 15, 3, step=2)
            
            if st.button(f"Apply {selected_filter}"):
                last_label, last_img = st.session_state['history'][-1]
                new_img = apply_filter(last_img, selected_filter, k_size)
                st.session_state['history'].append((f"+ {selected_filter}", new_img))
                st.rerun()

        st.divider()
        st.button("❌ Close / Reset All", on_click=reset_app, type="primary")

# --- 5. MAIN DISPLAY AREA ---

if not st.session_state['history']:
    st.info("👈 Please upload an image in the sidebar to get started.")
else:
    # Display logic: We show images side-by-side. 
    # If there are many, we rely on Streamlit's column wrapping or scroll.
    # To keep it "side by side" effectively, we use st.columns with the length of history.
    
    history_len = len(st.session_state['history'])
    
    # Create columns. We use a container to hold them.
    # Note: If history gets very long, columns get very thin. 
    # For a course demo, usually 3-5 steps are max, which fits fine.
    
    st.write(f"### Transformation Chain ({history_len} steps)")
    
    # We create a container for the visual gallery
    cols = st.columns(history_len)
    
    for idx, (label, img_arr) in enumerate(st.session_state['history']):
        with cols[idx]:
            # Highlight the last image
            if idx == history_len - 1:
                st.markdown(f"**Step {idx}: {label}** (Latest)")
                st.image(img_arr, use_container_width=True, channels="RGB", output_format="PNG")
            else:
                st.markdown(f"**Step {idx}: {label}**")
                st.image(img_arr, use_container_width=True, channels="RGB")