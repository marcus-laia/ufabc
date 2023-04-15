import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_graph(original, transformed, title, size=(5,3)):
    fig = plt.figure(figsize=size)

    plt.plot(original, transformed)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid("on")
    plt.title(title)

    return fig


def plot_matrix(mat, size=(3, 3)):
    fig = plt.figure(figsize=size)

    sns.heatmap(
        np.round(mat, decimals=2),
        annot=True if mat.shape[0] <= 9 else False,
        cmap='gray',
        alpha=1,
        linewidths=1,
        fmt='g',
        cbar=False
    )

    return fig


def plot_matrix_grid(matrices, shape, titles=[], size=None):
    rows, cols = shape

    if size is None:
        size = (10 * cols, 10 * rows)

    if len(titles) < len(matrices):
        titles = titles + (len(titles) - len(matrices)) * [""]

    fig = plt.figure(figsize=size)
    
    for i in range(len(matrices)):
        ax = plt.subplot(rows, cols, i + 1)
        sns.heatmap(
            np.round(matrices[i], 2),
            annot=True,
            cmap='gray',
            alpha=1,
            linewidths=1,
            fmt='g',
            cbar=False,
            annot_kws={"fontsize":30}
        )
        ax.set_title(titles[i])

    return fig


def plot_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.colorbar(label='Intensity')
    plt.axis("off")


def plot_images_grid_old(original, transformed, size=(24,12)):
    fig = plt.figure(figsize=size)
    
    plt.subplot(1,2,1)
    plot_image(original, "Original Image")

    plt.subplot(1,2,2)
    plot_image(transformed, "Transformed Image")

    return fig


def plot_images_grid(images, shape, titles=[], size=None):
    rows, cols = shape

    if size is None:
        size = (10 * cols, 10 * rows)

    if len(titles) < len(images):
        titles = titles + (len(titles) - len(images)) * [""]

    fig = plt.figure(figsize=size)
    
    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        plot_image(images[i], titles[i])

    return fig