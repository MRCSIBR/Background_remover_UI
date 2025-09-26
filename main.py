import streamlit as st
from PIL import Image
import io
import torch
from rembg import remove

# Set Streamlit page configuration
# Removed the 'icon' argument for wider compatibility
st.set_page_config(
    page_title="Background Remover App",
    layout="wide"
)

# Function to remove the background from a PIL Image
@st.cache_resource
def get_rembg_model():
    """
    This function initializes the rembg model.
    Using st.cache_resource ensures the model is loaded only once,
    even across reruns of the app, saving time and memory.
    """
    # It intelligently handles device (CPU/GPU) based on torch.cuda.is_available() and rembg[gpu] installation.
    return remove # We return the function directly, as it handles model loading internally on first call

def remove_background_from_pil_image(input_image_pil):
    """
    Removes the background from a PIL Image using rembg.

    Args:
        input_image_pil (PIL.Image.Image): The input image as a PIL Image object.

    Returns:
        PIL.Image.Image: The output image with the background removed.
    """
    # Check for CUDA availability for user feedback
    # Removed the 'icon' argument from st.info and st.warning for compatibility
    if torch.cuda.is_available():
        st.info("CUDA is available. Using GPU for background removal.")
    else:
        st.warning("CUDA not available. Using CPU for background removal, which may be slower.")

    # Get the rembg remove function (model loaded once via cache_resource)
    rembg_remove_func = get_rembg_model()

    # Perform background removal
    output_image = rembg_remove_func(input_image_pil)
    return output_image

# --- Streamlit App UI ---
st.title("✂️ Smart Background Remover")
st.markdown("### Drag and drop your image to instantly remove its background!")

# File uploader widget for drag-and-drop
uploaded_file = st.file_uploader(
    "Upload an image...",
    type=["png", "jpg", "jpeg", "webp"],
    help="Supports PNG, JPG, JPEG, and WebP formats. Drag and drop your file here.",
    accept_multiple_files=False
)

if uploaded_file is not None:
    # Display the original image and provide a placeholder for the processed image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Processing image... This might take a moment, especially if the model is downloading for the first time or running on CPU."):
        processed_image = remove_background_from_pil_image(original_image)

    with col2:
        st.subheader("Background Removed")
        st.image(processed_image, caption="Processed Image (Background Removed)", use_column_width=True)

    st.success("Background removed successfully! 🎉")

    # Provide a download button for the processed image
    buf = io.BytesIO()
    processed_image.save(buf, format="PNG") # Save as PNG to preserve transparency
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Processed Image (PNG)",
        data=byte_im,
        file_name="background_removed_image.png",
        mime="image/png",
        help="Click to download the image with its background removed."
    )
else:
    st.info("Waiting for an image to be uploaded. Drag and drop or click 'Upload an image...' above.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, PyTorch, and `rembg`.")
