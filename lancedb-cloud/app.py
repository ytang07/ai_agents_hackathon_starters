import streamlit as st
from PIL import Image
import base64
import io
from inference_utils import retrieve

st.set_page_config(layout="wide")
st.title("🏠 Multimodal dataset search")

# --- Sidebar for Inputs ---
st.sidebar.header("Search Criteria")
search_type = st.sidebar.radio("Search by:", ("Text", "Image"))

query = None
uploaded_file = None

if search_type == "Text":
    text_query = st.sidebar.text_input("Enter description of the image", "")
    if text_query:
        query = text_query
else:
    uploaded_file = st.sidebar.file_uploader("Upload an image of a house", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        encoded_string = base64.b64encode(image_bytes).decode()
        image_type = uploaded_file.type
        query = f"data:{image_type};base64,{encoded_string}"


limit = st.sidebar.number_input("Number of results to display:", min_value=1, max_value=100, value=9, step=1)

search_button = st.sidebar.button("Search")

if search_button and query:
    st.subheader(f"Search Results (Top {limit})")
    try:
        with st.spinner("Retrieving results..."):
            
            results_df = retrieve(query, limit=limit)

        if results_df is not None and not results_df.empty:
            num_columns = 4
            cols = st.columns(num_columns)
            
            target_image_size = (300, 300)

            for index, row in results_df.iterrows():
                image_bytes = row["image"]
                # Get all columns except 'image' as metadata
                metadata = {k: v for k, v in row.items() if k != "image"}
                
                col_index = index % num_columns
                with cols[col_index]:
                    if image_bytes and isinstance(image_bytes, bytes):
                        try:
                            img = Image.open(io.BytesIO(image_bytes))
                            img_resized = img.resize(target_image_size, Image.Resampling.LANCZOS)
                            st.image(img_resized, use_container_width=True)
                            
                            # Display metadata in expander
                            with st.expander("View metadata"):
                                for key, value in metadata.items():
                                    if value is not None and str(value).strip():
                                        st.markdown(f"**{key}**: {value}")
                        except Exception as e:
                            st.error(f"Could not display image: {e}")
                    else:
                        st.warning("Image data not available or in wrong format.")
                    st.markdown("---") 
        elif results_df is not None and results_df.empty:
            st.info("No results found for your query.")
        else:
            st.error("Failed to retrieve results. Check the logs.")
            
    except Exception as e:
        st.error(f"An error occurred during search: {e}")
        st.error("Please ensure your backend API is running and accessible, and the LanceDB table exists.")

elif search_button and not query:
    st.warning("Please enter a text query or upload an image.")

st.sidebar.markdown("---")
st.sidebar.markdown("Powered by LanceDB & GMI")