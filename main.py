import streamlit as st
from PIL import Image
import io
import torch
from rembg import remove

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Background Remover",
    layout="wide"
)

# --- Model Caching ---
@st.cache_resource
def get_rembg_model():
    """
    Load and cache the rembg model.
    The model is loaded only once and stored in cache for all subsequent runs.
    """
    # The 'remove' function from rembg handles model loading internally.
    return remove

# --- Core Processing Function ---
def remove_background(input_image_pil, session_state):
    """
    Removes the background from a PIL Image using the settings from the session state.

    Args:
        input_image_pil (PIL.Image.Image): The input image.
        session_state: The Streamlit session state containing the parameters.

    Returns:
        PIL.Image.Image: The output image with the background removed.
    """
    # Show a status message based on CUDA availability
    if torch.cuda.is_available():
        st.info("CUDA is available. Using GPU for processing.")
    else:
        st.warning("CUDA not available. Using CPU, which may be slower.")

    rembg_remove_func = get_rembg_model()

    # Pass the parameters from the session state to the rembg remove function
    processed_image = rembg_remove_func(
        input_image_pil,
        alpha_matting=session_state.alpha_mat,
        alpha_matting_foreground_threshold=session_state.fg_thresh,
        alpha_matting_background_threshold=session_state.bg_thresh,
        alpha_matting_erode_size=session_state.erode_size
    )
    return processed_image

# --- Sidebar UI for Settings ---
st.sidebar.header("⚙️ Fine-Tuning Settings")
st.sidebar.markdown(
    """
    Adjust these settings to improve the quality of the background removal,
    especially around complex edges like hair.
    """
)

# Initialize session state for widgets if they don't exist
# This is crucial for keeping the widget values persistent across reruns
if 'alpha_mat' not in st.session_state:
    st.session_state.alpha_mat = True
if 'fg_thresh' not in st.session_state:
    st.session_state.fg_thresh = 240
if 'bg_thresh' not in st.session_state:
    st.session_state.bg_thresh = 10
if 'erode_size' not in st.session_state:
    st.session_state.erode_size = 10

# Create the widgets and bind them to the session state
st.session_state.alpha_mat = st.sidebar.checkbox(
    "Enable Alpha Matting",
    value=st.session_state.alpha_mat,
    help="Refines the edges of the foreground, creating a smoother and more accurate cutout. Can be slower."
)

# Conditionally show alpha matting sliders only if the feature is enabled
if st.session_state.alpha_mat:
    st.session_state.fg_thresh = st.sidebar.slider(
        "Foreground Threshold", 0, 255, st.session_state.fg_thresh,
        help="Sets the threshold for identifying foreground pixels. Higher values are more strict. Default is 240."
    )
    st.session_state.bg_thresh = st.sidebar.slider(
        "Background Threshold", 0, 255, st.session_state.bg_thresh,
        help="Sets the threshold for identifying background pixels. Lower values are more strict. Default is 10."
    )
    st.session_state.erode_size = st.sidebar.slider(
        "Erode Size", 0, 30, st.session_state.erode_size,
        help="Shrinks the foreground mask to help remove small background artifacts around the edges. Default is 10."
    )


# --- Main Page UI ---
st.title("✂️ Advanced Background Remover")
st.markdown("### Upload an image and use the sidebar settings to fine-tune the result.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["png", "jpg", "jpeg", "webp"],
    help="Drag and drop your file here."
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    original_image = Image.open(uploaded_file)

    with col1:
        st.subheader("Original Image")
        st.image(original_image, caption="Uploaded Image", use_column_width=True)

    # When the user changes a slider, Streamlit re-runs the script.
    # The processing function is called with the new values from session_state.
    with st.spinner("Applying settings and removing background..."):
        processed_image = remove_background(original_image, st.session_state)

    with col2:
        st.subheader("Processed Image")
        st.image(processed_image, caption="Background Removed", use_column_width=True)

    st.success("Processing complete! 🎉")

    # Prepare image for download
    buf = io.BytesIO()
    processed_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Processed Image (PNG)",
        data=byte_im,
        file_name=f"bg_removed_{uploaded_file.name.split('.')[0]}.png",
        mime="image/png"
    )
else:
    st.info("Upload an image to get started.")

st.markdown("---")
st.markdown("Built with Streamlit, PyTorch, and the `rembg` library.")
