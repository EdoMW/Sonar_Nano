from __future__ import print_function
import cv2 as cv
import matplotlib.pyplot as plt
from numpy import *
from pyueye import ueye
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def ueye_take_picture_2(image_number):
    # Variables
    hCam = ueye.HIDS(0)  # 0: first available camera;  1-254: The camera with the specified camera ID
    sInfo = ueye.SENSORINFO()
    cInfo = ueye.CAMINFO()
    pcImageMemory = ueye.c_mem_p()
    MemID = ueye.int()
    rectAOI = ueye.IS_RECT()
    pitch = ueye.INT()
    nBitsPerPixel = ueye.INT(24)  # 24: bits per pixel for color mode; take 8 bits per pixel for monochrome
    channels = 3  # 3: channels for color mode(RGB); take 1 channel for monochrome
    m_nColorMode = ueye.INT()  # Y8/RGB16/RGB24/REG32
    bytes_per_pixel = int(nBitsPerPixel / 8)
    # ---------------------------------------------------------------------------------------------------------------------------------------
    # print("START")
    # print()
    # Starts the driver and establishes the connection to the camera
    nRet = ueye.is_InitCamera(hCam, None)
    # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
    nRet = ueye.is_GetCameraInfo(hCam, cInfo)
    # You can query additional information about the sensor type used in the camera
    nRet = ueye.is_GetSensorInfo(hCam, sInfo)
    # Set display mode to DIB
    nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
    # Set the right color mode
    if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
        # setup the color depth to the current windows setting
        ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
        bytes_per_pixel = int(nBitsPerPixel / 8)
    # Can be used to set the size and position of an "area of interest"(AOI) within an image
    nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
    # if nRet != ueye.IS_SUCCESS:
    #     print("is_AOI ERROR")
    width = rectAOI.s32Width
    height = rectAOI.s32Height

    # ---------------------------------------------------------------------------------------------------------------------------------------

    # Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
    nRet = ueye.is_AllocImageMem(hCam, width, height, nBitsPerPixel, pcImageMemory, MemID)
    if nRet != ueye.IS_SUCCESS:
        print("is_AllocImageMem ERROR")
    else:
        # Makes the specified image memory the active memory
        nRet = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_SetImageMem ERROR")
        else:
            # Set the desired color mode
            nRet = ueye.is_SetColorMode(hCam, m_nColorMode)

    # Activates the camera's live video mode (free run mode)
    nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
    if nRet != ueye.IS_SUCCESS:
        print("is_CaptureVideo ERROR")

    nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Continuous image display
    if nRet == ueye.IS_SUCCESS:
        # In order to display the image in an OpenCV window we need to...
        # ...extract the data of our image memory
        pic_array_1 = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
        time.sleep(0.4)
        pic_array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)

        # ...reshape it in an numpy array...
        frame = np.reshape(pic_array, (height.value, width.value, bytes_per_pixel))
        frame = np.pad(frame, pad_width=[(506, 506), (0, 0), (0, 0)], mode='constant') # pad with zeros above and under
        # ...resize the image by a half
        frame = cv.resize(frame, (0, 0), fx=0.331606, fy=0.331606)

        # ---------------------------------------------------------------------------------------------------------------------------------------

        t = time.localtime()
        current_time = time.strftime("%H_%M_%S", t)
        frame = frame[:,:,0:3]
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # TODO (after exp) make folder_path_for_images a parameter
        folder_path_for_images = r'D:\Users\NanoProject\Images_for_work'
        img_name = 'num_dt.jpeg'
        img_name = img_name.replace("num", str(image_number))
        img_name = img_name.replace("dt", str(current_time))
        image_path = os.path.join(folder_path_for_images, img_name)
        plt.imsave(image_path, frame)
        cv.destroyAllWindows()
    # ---------------------------------------------------------------------------------------------------------------------------------------
    else:
        print("no image was taken")
    # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
    ueye.is_FreeImageMem(hCam, pcImageMemory, MemID)

    # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
    ueye.is_ExitCamera(hCam)

    # Destroys the OpenCv windows
    cv.destroyAllWindows()
    #
    # print()
    # print("END")
    return image_path, frame


a,b = ueye_take_picture_2(2)
cv.imshow("check", b)
cv.waitKey()

# Facing West
#image 36 (DSC_0036.JPG), NIKON  2_11_04_40 uEye
# Facing East
# image 42 (DSC_0036.JPG), NIKON  2_11_04_40 uEye

