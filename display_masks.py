import os
import random
import colorsys
import cv2

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon
from self_utils import utils


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def show_in_moved_window_1(win_name, img, i=None, x=0, y=0, wait_time=0):  # lab
    """
    show image
    :param wait_time: time to wait before stop showing image
    :param win_name: name of the window
    :param img: image to display
    :param i: index of the grape
    :param x: x coordinate of end left corner of the window
    :param y: y coordinate of end left corner of the window
    """
    if img is not None:
        target_bolded = img.copy()
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)  # Create a named window
        cv2.moveWindow(win_name, x, y)  # Move it to (x,y)
        cv2.imshow(win_name, target_bolded)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    if N == 1:
        scores = np.array([scores])
    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    # print('masked_image', type(masked_image), masked_image.shape)
    # rgb_masked_image = cv2.cvtColor(masked_image.astype(np.uint8).copy(), cv2.COLOR_RGB2BGR)
    show_in_moved_window_1('image with masks!!!!', masked_image.astype(np.uint8), None, 0, 0, 0)
    # cv2.imshow("image with masks!!!!", masked_image.astype(np.uint8))
    # cv2.moveWindow("image with masks", x=(-1040), y=(-5))
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    if auto_show:
        plt.show()
    return masked_image.astype(np.uint8)

def load_mask_file_1(image_number):
    directory = r'old_experiments\exp_data_13_46'
    parent_dir = r'D:\Users\NanoProject'
    path = os.path.join(parent_dir, directory)
    path = os.path.join(path, 'masks_2')
    records_list = os.listdir(path)
    res = [i for i in records_list if i.startswith(str(image_number) + "_")]
    assert len(res) > 0, 'list is empty. no mask detected on this frame'
    path = os.path.join(path, res[0])
    mask = np.load(path)
    mask = mask.f.arr_0
    return mask


def load_image(image_number):
    """
    :return: path for image to be loaded
    """
    directory = r'old_experiments\exp_data_13_46'
    parent_dir = r'D:\Users\NanoProject'
    path = os.path.join(parent_dir, directory)
    path = os.path.join(path, 'rgb_images')
    path = os.path.join(path, 'resized')
    locations_list = os.listdir(path)
    res = [i for i in locations_list if i.startswith(str(image_number) + "_")]
    path = os.path.join(path, res[0])
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


for i in range(2):
    masks = load_mask_file_1(i)
    image = load_image(i)
    boxes = utils.extract_bboxes(masks).astype('int32')
    class_ids = np.array([1] * masks.shape[2])
    class_names = ['BG', 'grape_cluster']
    display_instances(image, boxes, masks, class_ids, class_names)