import streamlit as st
from intensitytransformation import IntensityTransformation
from filtering import Filtering

def build_sidebar():
    page = st.sidebar.selectbox("Select", ("Intensity Transformation", "Filtering"))

    return page

if __name__ == "__main__":
    st.title("Medical Image Processing")

    page = build_sidebar()

    if page == "Intensity Transformation":
        page_builder = IntensityTransformation()
    elif page == "Filtering":
        page_builder = Filtering()

    page_builder.build_page()