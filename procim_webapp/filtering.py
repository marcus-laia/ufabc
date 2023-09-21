import cv2 as cv
import numpy as np
import streamlit as st

from utils.plotting_utils import *

class Filtering():
    def __init__(self) -> None:
        self.image_name = None
        self.filter_name = None
        self.kernel_size = None

    def build_sidebar(self):
        self.image_name = st.sidebar.selectbox("Select Image", (None, "Camera"))
        filter_type = st.sidebar.selectbox("Select Filter", ("LowPass", "HighPass"))

        if filter_type == "LowPass":
            spec_options = ("Mean", "Median")
        elif filter_type == "HighPass":
            spec_options = ("Default1", "Default2", "Vertical", "Horizontal", "Sobel")

        filter_spec = st.sidebar.selectbox("Filter Specification", spec_options)

        if filter_type == "LowPass":
            self.kernel_size = st.sidebar.slider("Kernel Size", 3, 25, 3, 2)
        else:
            self.kernel_size = 3

        self.filter_name = filter_type + filter_spec

    def apply_median_filter(self, img):
        result = np.zeros(img.shape)
        step_back = int(np.floor(self.kernel_size / 2))
        step_foward = int(np.ceil(self.kernel_size / 2))

        for row in range(img.shape[0]):
            start_row = min(0, row - step_back)
            end_row = max(row + step_foward, img.shape[0])
            for col in range(img.shape[1]):
                start_col = min(0, col - step_back)
                end_col = max(col + step_foward, img.shape[0])

                result[row][col] = np.round(np.median(img[start_row : end_row, start_col : end_col]))

        return result
    
    def get_kernel(self):
        if self.filter_name == "LowPassMean":
            return np.ones((self.kernel_size, self.kernel_size)) / (self.kernel_size**2)
        elif self.filter_name == "HighPassDefault1":
            return np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        elif self.filter_name == "HighPassDefault2":
            return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        elif self.filter_name == "HighPassVertical":
            return np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        elif self.filter_name == "HighPassHorizontal":
            return np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        elif self.filter_name == "HighPassSobel":
            return [
                np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
                np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            ]
        return np.ones((self.kernel_size, self.kernel_size))

    @staticmethod
    def pad_kernel(kernel, shape):
        diff_rows = shape[0] - kernel.shape[0]
        diff_cols = shape[1] - kernel.shape[1]

        padded = np.pad(kernel, [[0, diff_rows], [0, diff_cols]])
        return padded
    
    @staticmethod
    def _get_shift_magnitude(mat_freq):
        mat_shift = np.fft.fftshift(mat_freq)
        mat_shift_mag = 20 * np.log(abs(mat_shift))
        return mat_shift_mag

    def to_frequency_domain(self, mat, get_view=True):
        mat_freq = np.fft.fft2(mat)

        if get_view:
            mat_shift_mag = self._get_shift_magnitude(mat_freq)
        else:
            mat_shift_mag = None

        return mat_freq, mat_shift_mag

    def filter_image_freq(self, img, kernel):
        img_freq, img_freq_view = self.to_frequency_domain(img)

        padded_kernel = self.pad_kernel(kernel, img.shape)
        kernel_freq, _ = self.to_frequency_domain(padded_kernel, get_view=False)

        filtered_img_freq = img_freq * kernel_freq
        filtered_img_freq_view = self._get_shift_magnitude(filtered_img_freq)

        filtered_img = np.real(np.fft.ifft2(filtered_img_freq))

        return [filtered_img, img_freq_view, filtered_img_freq_view]
    
    def apply_filter(self, img, kernel):
        if self.filter_name == "LowPassMedian":
            # result = [self.apply_median_filter(img)]
            result = [cv.medianBlur(img, ksize=self.kernel_size)]
        elif self.filter_name == "HighPassSobel":
            filtered_x, _, _ = self.filter_image_freq(img, kernel[0])
            filtered_y, _, _ = self.filter_image_freq(img, kernel[1])

            filtered = np.sqrt(np.square(filtered_x) + np.square(filtered_y))
            filtered *= 255.0 / filtered.max()

            result = [filtered, filtered_x, filtered_y]
        else:
            result = self.filter_image_freq(img, kernel)

        return result

    def get_kernel_plot(self, kernel):
        if self.filter_name == "HighPassSobel":
            fig = plot_matrix_grid(
                matrices=kernel,
                shape=(1,2),
                titles=["Horizontal Kernel", "Vertical Kernel"],
            )
        else:
            fig = plot_matrix(kernel)
        
        return fig
    
    def get_images_plot(self, images):
        if self.filter_name == "LowPassMedian":
            fig = plot_images_grid(images, (1, 2), ["Original", "Filtered"])
        else:
            if self.filter_name == "HighPassSobel":
                titles = ["Original", "Filtered", "Filtered Horizontal", "Filtered Vertical"]
            else:
                titles = ["Original", "Filtered", "Original Freq Domain", "Filtered Freq Domain"]

            fig = plot_images_grid(
                images=images,
                shape=(2, 2),
                titles=titles
            )
        
        return fig

    def build_page(self):
        self.build_sidebar()

        kernel = self.get_kernel()

        kernel_plot = self.get_kernel_plot(kernel)
        st.subheader("Filter Kernel(s)")
        st.pyplot(kernel_plot)

        if self.image_name is not None:
            img = cv.imread('./images/' + self.image_name.lower() + '.png', 0)
            results = self.apply_filter(img, kernel)
            images_grid = self.get_images_plot([img] + results)

            st.subheader("Results")
            st.pyplot(images_grid)
