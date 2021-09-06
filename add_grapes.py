import numpy as np
import cv2
import os
import sys
ROOT_DIR = os.path.abspath("C:/Drive/Mask_RCNN-master/")
sys.path.append(ROOT_DIR)
from mrcnn import utils

# screen_x_cord = -1_200  # TODO: when working on my computer. change to 200 when working in lab
# screen_y_cord = -120  # TODO: when working on my computer. change to 200 when working in lab
screen_x_cord = 70  # TODO: when working on my computer. change to 200 when working in lab
screen_y_cord = 0  # TODO: when working on my computer. change to 200 when working in lab
img_path = r'1.jpg'
output_mask_path = r'1.npy'


def read_image(rgb_image):
    """
    read 6000/4000/3 rgb image. resize it. return parameters to later resize the mask to the original size.
    :return: image_resized, scale_ratio, padding_dim
    """
    # image = cv2.imread(img_path)
    # print(a.shape)
    # a1 = cv2.resize(a, (1024, 1024))
    image = rgb_image
    image_resized, _, scale_ratio, padding_dim, _ = utils.resize_image(image)
    # print("scale", scale)
    # print("padding", padding)
    # print(type(a[0]), type(a1))
    return image_resized, scale_ratio, padding_dim


def show_img(title, image_to_show, x_cord, y_cord):
    cv2.imshow(title, image_to_show)
    cv2.moveWindow(title, x_cord, y_cord)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


def get_rect_box(cnt):
    hull = cv2.convexHull(cnt)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return rect, box


class DrawLineWidget(object):
    """
    The algorithm that enable draw new masks and calculate OBB's.
    Mouse clicks:
    left button click- add new point. each click added (after the first one) will draw a line between them.
    if click was done close enough to the first (red) point, the mask will be saved. at least 3 points are required.
    right button click- clear the last mask that was in process.
    toggle scroll - erase the last point that was added (line/lines to be deleted marked in blue).
    Enter/Esc button- close the polygon with straight line between the first and last points.
    """
    def __init__(self, img_shared):
        self.original_image = img_shared
        self.clone = img_shared.copy()
        self.clone_2 = self.clone.copy()
        cv2.namedWindow('Add grape mask manually')
        cv2.setMouseCallback('Add grape mask manually', self.extract_coordinates)
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, *_):
        # Record starting (x,y) coordinates on left mouse button click (next two if, elif)
        if event == cv2.EVENT_LBUTTONDOWN:
            try:
                self.clone_2 = self.clone.copy()
            except Exception as e:
                print("exception :", e.__class__)
            self.image_coordinates.append((x, y))
            x_start, y_start = self.image_coordinates[0][0], self.image_coordinates[0][1]
            if len(self.image_coordinates) > 1:
                if abs(((x - x_start) ** 2)+((y - y_start) ** 2)) < 75:
                    # print("finished")
                    cv2.destroyAllWindows()
        elif event == cv2.EVENT_LBUTTONUP:
            # self.image_coordinates.append((x,y))
            len_line = len(self.image_coordinates)
            # print(f'Amount of verices: {len_line}', "mouse up", self.image_coordinates)
            self.clone = self.original_image.copy()
            if len_line > 0:
                cv2.circle(self.clone, (int(self.image_coordinates[0][0]), int(self.image_coordinates[0][1])),
                           radius=3, color=(12, 36, 255), thickness=3)
                cv2.circle(self.clone, (int(self.image_coordinates[-1][0]), int(self.image_coordinates[-1][1])),
                           radius=2, color=(36, 255, 12), thickness=3)
            if len_line > 1:
                for j in range(len(self.image_coordinates) - 1):
                    cv2.line(self.clone, self.image_coordinates[j], self.image_coordinates[j + 1], (36, 255, 12), 2)
                # cv2.line(self.clone, self.image_coordinates[len_line-2],
                # self.image_coordinates[len_line-1], (36,255,12), 2)
            show_img(title="Add grape mask manually", image_to_show=self.clone, x_cord=screen_x_cord, y_cord=screen_y_cord)
            # self.clone_2 = self.clone.copy()
        # Clear drawing boxes on right mouse button click (next two elif)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()
            self.image_coordinates = []
        elif event == cv2.EVENT_RBUTTONUP:
            # cv2.imshow("image", self.clone)
            show_img(title="Add grape mask manually", image_to_show=self.clone, x_cord=screen_x_cord, y_cord=screen_y_cord)
        # Clear last point added on mouse wheel scroll
        elif event == cv2.EVENT_MOUSEWHEEL:
            cv2.line(self.clone_2, self.image_coordinates[-2], self.image_coordinates[-1], (255, 36, 12), 2)
            self.image_coordinates.pop()
            # print(self.image_coordinates)
            # cv2.imshow("image", self.clone_2)
            show_img(title="Add grape mask manually", image_to_show=self.clone_2, x_cord=screen_x_cord, y_cord=screen_y_cord)
            # self.clone_2 = self.clone.copy()
        elif event == cv2.EVENT_MOUSEMOVE:
            if len(self.image_coordinates) > 2:
                x_start, y_start = self.image_coordinates[0][0], self.image_coordinates[0][1]
                if abs(((x - x_start) ** 2)+((y - y_start) ** 2)) < 175:
                    cv2.circle(self.clone, (int(self.image_coordinates[0][0]), int(self.image_coordinates[0][1])),
                               radius=7, color=(120, 36, 255), thickness=3)
            pass

    def show_image(self):
        return self.clone


def get_unrecognized_grapes(img_t, amount_of_grapes):
    """
    Let you draw new grapes there were not detected by the network.
    for use during the experiment.
    :param img_t: image with current grape masks that were detected.
    :param amount_of_grapes: amount of new grapes to be added
    :return: obbs, corners, npys, img_shared
    """
    img_shared = img_t.copy()
    obbs, npys, corners = [], [], []

    image_index = 0
    # amount_of_grapes = int(input("enter amount of grapes: "))
    # amount_of_grapes = 2
    while image_index < amount_of_grapes:
        draw_line_widget = DrawLineWidget(img_shared)
        while True:
            show_img(title='Add grape mask manually', image_to_show=draw_line_widget.show_image(),
                     x_cord=screen_x_cord, y_cord=screen_y_cord)
            if cv2.waitKey() or 0xFF == 27:
                break
        cv2.destroyAllWindows()
        cv2.waitKey(0) & 0xFF
        array = draw_line_widget.image_coordinates  # no predefined polygon
        if len(array) < 3:
            continue
        image_index += 1
        print(f'{image_index}/{amount_of_grapes} masks were added so far.')

        img = img_t.copy()
        for i in range(len(array) - 1):
            cv2.line(img, array[i], array[i + 1], (36, 255, 12), 2)
        cv2.line(img, array[0], array[i + 1], (36, 255, 12), 2)
        cv2.destroyAllWindows()
        array = [list(ele) for ele in array]
        array = np.array([np.array(xi) for xi in array])
        points = array
        for i in range(len(points)):
            cv2.circle(img, (int(points[i][0]), int(points[i][1])), radius=4, color=(12, 12, 255),
                       thickness=3)
            # cv2.putText(img, f'{(int(points[i][0]), int(points[i][1]))}',
            #             org=(int(points[i][0]) - 75, int(points[i][1]) + 35),
            #             fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.51,
            #             color=(255, 255, 255), thickness=1, lineType=2)
        img_1 = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
        img_1.fill(255)

        cv2.polylines(img_1, [array], True, (0, 15, 255))
        cv2.drawContours(img_1, [array], -1, (0, 15, 255), -1)
        src_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        del img_1
        ret, thresh = cv2.threshold(src_gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        rect, box = get_rect_box(contours[0])
        npy = np.asarray(thresh, dtype=np.uint8)
        img_shared = cv2.drawContours(img_shared, [box], 0, (10, 154, 255), 2)
        # if image_index == amount_of_grapes: # optional - indent lines 141-151 and add the if.
        try:
            img_shared_overlay = img_shared.copy()
        except Exception as e:
            print("exception :", e.__class__)
        cv2.polylines(img_shared_overlay, [array], True, (0, 255, 255))
        cv2.fillPoly(img_shared_overlay, [array], 255)
        alpha = 0.4
        cv2.addWeighted(img_shared_overlay, alpha, img_shared, 1 - alpha,
                        0, img_shared)
        show_img(title=f"Add grape mask manually {image_index + 1}", image_to_show=img_shared,
                 x_cord=screen_x_cord, y_cord=screen_y_cord)
        # cv2.imshow()
        cv2.waitKey()
        cv2.destroyAllWindows()
        obbs.append(rect)
        corners.append(box.tolist())
        npys.append(npy)

    return obbs, corners, npys, img_shared


def add_grapes(rgb_image):

    mask, obbs_list, corners_list, npys_list = None, [], [], []
    img_rgb, scale, padding = read_image(rgb_image)
    while True:
        check_for_more = input(" \n More grapes to add?" '\n'
                               "if None press ENTER to continue. Else enter positive number: ")
        if check_for_more == "":
            break
        elif check_for_more.isdigit():
            obbs_temp, corners_list_temp, npys_temp, img_rgb = get_unrecognized_grapes(img_rgb, int(check_for_more))
            for grapes_added in range(int(check_for_more)):
                obbs_list.append(obbs_temp[grapes_added])
                corners_list.append(corners_list_temp[grapes_added])
                npys_list.append(npys_temp[grapes_added])
    print(f'total of {len(npys_list)} grapes were added manually')
    # if there were grapes to be added to the Data Base.
    if len(npys_list) > 0:
        # converting the array to np array in shape of [1024, 1024, N]
        npy_array = npys_list[0]
        if len(npys_list) > 1:
            for arr in range(1, len(npys_list)):
                npy_array = np.dstack((npy_array, npys_list[arr]))
        else:
            npy_array = np.reshape(npy_array, (1024, 1024, 1))
        mask = npy_array
        # TODO: next 5 lines are for resize and save mask back to the original image size.
        # npy_array = npy_array[170:853, :, :]  # remove the padding
        # mask = utils.resize_mask(npy_array, 1/scale, padding=0, crop=None)  # make it [4002,6000,N]
        # mask = mask[1:4001, :, :]  # fix the resizing
        # mask = mask.clip(max=1)  # fix colors in visualization module
        # np.save(file=output_mask_path, arr=mask)  # save the name mask.
    return mask, obbs_list, corners_list, img_rgb
