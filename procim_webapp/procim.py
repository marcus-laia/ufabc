import streamlit as st
from intensitytransformation import IntensityTransformation

def build_sidebar():
    page = st.sidebar.selectbox("Select", ("Intensity Transformation",))

    return page

if __name__ == "__main__":
    st.title("Medical Image Processing")

    page = build_sidebar()

    if page == "Intensity Transformation":
        page_builder = IntensityTransformation()

    page_builder.build_page()