import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors as mcolors
import numpy as np

def plot_batch(batch_data):
    images, gts = batch_data
    batch_size = len(images)
    frame_count = len(images[0])

    images = images.numpy()

    plt.ion()
    f, axs = plt.subplots(batch_size, frame_count)

    for i in range(batch_size):
        for j in range(frame_count):
            image = images[i][j]
            
            # Tensor shape in [C, H, W]
            c, h, w = image.shape
            image_size = np.array([w, h, w, h, 1])

            image = np.transpose(image, (1, 2, 0))
            
            axs[i][j].imshow(image)
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            
            gt = gts[i][j]
            for k, bbox in enumerate(gt):
                plot_bbox(axs[i][j], bbox.numpy() * image_size, 'r')
            
    plt.tight_layout()
    plt.ioff()

    plt.show()

def plot_result_batch(batch_data):
    images, results = batch_data
    bboxes, labels = results

    batch_size = len(images)
    frame_count = len(images[0])

    images = images.numpy()

    plt.ion()
    f, axs = plt.subplots(batch_size, frame_count)

    for i in range(batch_size):
        for j in range(frame_count):
            image = images[i][j]
            image = np.transpose(image, (1, 2, 0))
            
            axs[i][j].imshow(image)
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            
            frame_bboxes = bboxes[i][j]
            frame_labels = labels[i][j]
            show_bboxes(axs[i][j], frame_bboxes, frame_labels, None)
            
    plt.tight_layout()
    plt.ioff()

    plt.show()

def plot_bbox(ax, bbox, color):
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=1,
        edgecolor=color,
        facecolor='none'
    )
    ax.add_patch(rect)

    return rect

def show_bboxes(ax, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'k'])

    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]

        rect = plot_bbox(ax, bbox, color)

        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            ax.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
