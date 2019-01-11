import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy
import numpy as np


class RegionProposals:
    def __init__(self):
        pass

    def selective_search(self, img, plot_image=False):
        img_lbl, regions = selectivesearch.selective_search(img, scale=200, sigma=0.9, min_size=100)
        candidates = set()
        for r in regions:
            x, y, w, h = r['rect']
            candidates.add(r['rect'])
        if plot_image:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
            ax.imshow(img)
            plt.title('Selective search boxes')
            for x, y, w, h in candidates:
                rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=3)
                ax.add_patch(rect)
            plt.show()
        return candidates

    def intersection_over_union(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def get_xy(self, box):
        x, y, w, h = box
        x_top_left = x
        y_top_left = y
        x_right_bottom = x_top_left + w
        y_right_bottom = y_top_left + h
        return [x_top_left, y_top_left, x_right_bottom, y_right_bottom]

    def get_region_of_proposals(self, image, plot_regions= True):
        candidates = self.selective_search(image, True)
        selected_regions = list()
        boxA = [0, 0, image.shape[0], image.shape[1]]
        for candidate in candidates:
            boxB = self.get_xy(candidate)
            iou = self.intersection_over_union(boxA, boxB)
            if iou >= 0.5:
                selected_regions.append(candidate)
        # resize according to alexnet input and return selected regions
        images = np.array([scipy.misc.imresize(image[r[0]:r[3], r[1]:r[2], :], (227, 227)) for r in selected_regions])
        if plot_regions:
            fig, axes = plt.subplots(1, images.shape[0], figsize=(20, 20))
            for i, ax in enumerate(axes.flat):
                ax.imshow(images[i])
                ax.set_xticks([])
                ax.set_yticks([])
            plt.title('Region Of Proposals')
            plt.show()
        return images / 255
