from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """

        # Negative is steering to the right 
    left_bias = -1
    steer_matrix_left = np.zeros(shape = shape, dtype = np.int8)
    steer_matrix_left[:, :int(shape[1]/2)] = left_bias
    # ---
    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """

    steer_matrix_right = np.zeros(shape = shape, dtype = np.int8)
    steer_matrix_right[: , int(shape[1]/2):] = 1
    # ---
    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        left_masked_img:   Masked image for the dashed-yellow line (numpy.ndarray)
        right_masked_img:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape
    
    # Convert the image to other useful forms
    img = convert_image_forms(image)[0]
    imghsv = convert_image_forms(image)[1]
    
    # Masks 
        # Gaussian Blurring 
    img_gaussian_filter = gaussain_blur(img)
        # Sobel Edge Detection: 
    mask_mag = sobel_edge_detection(img_gaussian_filter)[0]
    sobelx = sobel_edge_detection(img_gaussian_filter)[1]
    sobely = sobel_edge_detection(img_gaussian_filter)[2]

        # Colour Masking 
    mask_white = colour_masking(imghsv)[0]
    mask_yellow = colour_masking(imghsv)[1]

        # Edge Based marking
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(w/2)): (w + 1)] = 0
    mask_right = np.ones(sobely.shape)
    mask_right[:, 0:int(np.floor(w/2))] = 0

    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    mask_left_edge = (mask_mag*mask_left*mask_sobelx_neg*mask_sobely_neg*mask_yellow) 
    # Inner edge pf dashed yellow line corresponds to neg grad in and y  

    mask_right_edge = (mask_mag*mask_right*mask_sobelx_pos*mask_sobely_neg*mask_white) # Inner edge of solid white line represents a pos grad in x and neg grad in y 

    return mask_left_edge, mask_right_edge

# Helper Functions 

def convert_image_forms(image: np.ndarray) -> np.ndarray:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray) 

    Return:
        imghsv: Image converted to HSV space 
        img: Image converted to gray space for processing
    """
    # Conver the image to HSV for any colour-based filtering 
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return img, imghsv

def gaussain_blur(image: np.array) -> np.ndarray:
    """

    Args: 
        image: An image in gray space 

    Return: 
        img_gaussian_filter: Image convolved with a guassian blur kernel
    """
    sigma = 4 
    img_gaussian_filter = cv2.GaussianBlur(image, (0,0) , sigma)

    return img_gaussian_filter

def sobel_edge_detection(img_gaussian_filter: np.ndarray ) -> np.ndarray:
    """
    Convolve the image with the sobel operator to compute the numerical derivatives in the x and y directions. Apply a non-maximal supression technique to the magnitude of the change in derivatives to create a map showing large derivative changes. 

    Args:
        img_gaussian_filter: An image from the robot's camera that has filtered with a gaussian filter (numpy.ndarray)

    Return: 
        mask_mag: A map of region with change derivative larger than threshold (numpy.ndarray)
    """
    sobelx = cv2.Sobel(img_gaussian_filter, cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter, cv2.CV_64F,0,1)
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)  # Compute the magnitude of the gradients 
    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True) # Compute the orientation of the gradients 

        # Non-maximal Supression
    threshold = 25
    mask_mag = (Gmag > threshold)

    return mask_mag, sobelx , sobely

def colour_masking(imghsv: np.ndarray) -> np.ndarray:
    """
    Args:
        imghsv: An image from the robot's camera converted to HSV space (numpy.ndarray)

    Return: 
        mask_white: Mask for image based on HSV regions for white 
        mask_yellow: Mask for image based on HSV regions for yellow
    """
    white_lower_hsv = np.array([0, 0, 190])         
    white_upper_hsv = np.array([179, 70, 255])   
    yellow_lower_hsv = np.array([15, 110, 126])        
    yellow_upper_hsv = np.array([37, 220, 255])  

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    return mask_white, mask_yellow

def rescale(a: float, U:float, L:float): 
    if np.allclose(L,U):
        return 0.0
    return (a-l)/(L-U)
