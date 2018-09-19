import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors as mcolors
import numpy as np

def plot_batch(images, gts, rows):
    batch_size = len(images)
    if batch_size // rows < batch_size / rows:
        cols = batch_size // rows + 1
    else:
        cols = batch_size // rows

    images = images.numpy()

    plt.ion()
    f, axs = plt.subplots(rows, cols)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < batch_size:
                image = images[idx]
            
                # Tensor shape in [C, H, W]
                c, h, w = image.shape
                image_size = np.array([w, h, w, h])

                image = np.transpose(image, (1, 2, 0))
            
                axs[i][j].imshow(image.squeeze(), cmap='gray')
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])

                bboxes = gts[0][idx]
                show_bboxes(axs[i][j], bboxes.numpy() * image_size)
            
    plt.tight_layout()
    plt.ioff()

    max_window() 

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

def plot_classification(images, gts, rows):
    batch_size = len(images)
    if batch_size // rows < batch_size / rows:
        cols = batch_size // rows + 1
    else:
        cols = batch_size // rows

    images = images.numpy()
    # gts = gts.numpy()

    plt.ion()
    f, axs = plt.subplots(rows, cols)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j

            if idx < batch_size:
                image = images[idx]
            
                # Tensor shape in [C, H, W]
                c, h, w = image.shape
                image_size = np.array([w, h, w, h])

                image = np.transpose(image, (1, 2, 0))
            
                axs[i][j].imshow(image.squeeze(), cmap='gray')
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])

                axs[i][j].set_title('{}'.format(gts[idx]))
            
    plt.tight_layout()
    plt.ioff()

    max_window()

    plt.show()

def plot_detection(images, detections, ws, hs, ids, rows):
    batch_size = len(images)
    if batch_size // rows < batch_size / rows:
        cols = batch_size // rows + 1
    else:
        cols = batch_size // rows

    images = images.numpy()

    plt.ion()
    f, axs = plt.subplots(rows, cols)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j

            if idx < batch_size:
                image = images[idx]

                w = ws[idx]
                h = hs[idx]
                title = ids[idx]

                # draw image and title
                image = np.transpose(image, (1, 2, 0))
            
                axs[i][j].imshow(image.squeeze(), cmap='gray')
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])

                axs[i][j].set_title('{}'.format(title))

                # draw bboxes
                # problem specific - we have only one foreground class
                dets = detections[idx, 1, :]

                # only keep score > 0.
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)

                if dets.dim() != 0:
                    boxes = dets[: 1:]
                    boxes[:, 0] *= w
                    boxes[:, 2] *= w
                    boxes[:, 1] *= h
                    boxes[:, 3] *= h
                    scores = dets[:, 0].cpu().numpy()

                    show_bboxes(
                        axs[i][j],
                        boxes,
                        ['{:.2f}'.format(s) for s in scores]
                    )

    plt.tight_layout()
    plt.ioff()

    max_window()

    plt.show()

def plot_scatter(xs, ys):
    plt.scatter(xs, ys)

    plt.xlabel('x')
    plt.ylabel('y')

    max_window()

    plt.show()

# insert this before plt.show()
def max_window():
    backend = plt.get_backend()

    if backend == 'TkAgg':
        mng = plt.get_current_fig_manager()
        ### works on Ubuntu??? >> did NOT working on windows
        # mng.resize(*mng.window.maxsize())
        mng.window.state('zoomed') #works fine on Windows!
    elif backend == 'wxAgg':
        mng = plt.get_current_fig_manager()
        mng.frame.Maximize(True)
    elif backend == 'Qt4Agg' or backend == 'Qt5Agg':
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
