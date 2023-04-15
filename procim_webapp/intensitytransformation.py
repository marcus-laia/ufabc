import imageio
import numpy as np
import streamlit as st
from utils.plotting_utils import *

class IntensityTransformation():
    def __init__(self) -> None:
        self.image = None
        self.transf_type = None

    @staticmethod
    def build_sidebar():
        image = st.sidebar.selectbox("Select Image", (None, "PIG_CT"))
        transf = st.sidebar.selectbox("Select Transformation", ("Sigmoid", "Ramp", "Invert"))

        return image, transf

    @staticmethod
    def build_sidebar_sigmoid(img_min, img_max):
        # slider: label, min, max, value, step
        mid = (img_min+img_max)//2
        n_bits = st.sidebar.slider("Num Bits", 3, 12, 8, 1)

        max_out = 2 ** n_bits - 1
        middle = st.sidebar.slider("Window Middle", img_min, img_max, mid, 1)
        width = st.sidebar.slider("Window Width", 1, 100, 20, 1)

        params = {'max_out': max_out, 'middle': middle, 'width': width}

        return params

    @staticmethod
    def build_sidebar_ramp(img_min, img_max):
        # slider: label, min, max, value, step
        n_bits = st.sidebar.slider("Num Bits", 3, 12, 8, 1)

        max_out = 2 ** n_bits - 1
        wmin = st.sidebar.slider("Window Min", img_min, img_max, img_min, 1)
        wmax = st.sidebar.slider("Window Max", img_min, img_max, img_max, 1)

        params = {'max_out': max_out, 'window_min': wmin, 'window_max': wmax}

        return params

    @staticmethod
    def build_sidebar_invert(*args, **kwargs):
        return {}

    @staticmethod
    def sigmoid(image, max_out, middle, width):
        sigmoid_image = max_out / (1.0 + np.exp(-(image - middle) / width))
        return sigmoid_image

    @staticmethod
    def ramp(image, max_out, window_min, window_max):
        coef = max_out / (window_max - window_min)
        ramp_image = coef * (image - window_min)
        ramp_image = np.clip(ramp_image, a_min=0, a_max=max_out)
        return ramp_image

    @staticmethod
    def invert(image):
        max_value = image.max()    
        inverted_image = max_value - image
        return inverted_image

    def build_page(self):
        image, transf_type = self.build_sidebar()

        if image is None:
            min_value, max_value = 0, 255
        else:
            img = imageio.imread('./images/' + image)
            min_value, max_value = img.min(), img.max()

        vec = np.array(range(min_value, max_value + 1))

        transf_params = getattr(self, f"build_sidebar_{transf_type.lower()}")
        params = transf_params(int(min_value), int(max_value))

        transf_exec = getattr(self, transf_type.lower())
        transformed_vec = transf_exec(vec, **params)

        graph = plot_graph(vec, transformed_vec, f"{transf_type} applied ({image = })")

        st.subheader("Transformation Function")
        st.pyplot(graph)

        if image is not None:
            transformed_img = transf_exec(img, **params)

            images_grid = plot_images_grid(
                images=(img, transformed_img),
                shape=(1, 2),
                titles=("Original Image", "Transformed Image")
            )

            st.subheader("Transformed Image")
            st.pyplot(images_grid)
