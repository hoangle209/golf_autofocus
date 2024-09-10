import numpy as np
import cv2

from PIL import ImageFilter, ImageStat
from PIL import Image

# import pywt

# def rgb_to_hsi(image):
#     image = image.astype(np.float32) / 255.0
#     R, G, B = cv2.split(image)
    
#     # Calculate intensity
#     I = (R + G + B) / 3.0
    
#     # Calculate saturation
#     min_RGB = np.minimum(np.minimum(R, G), B)
#     S = 1 - (min_RGB / (I + 1e-6))  # Avoid division by zero
    
#     # Calculate hue
#     num = 0.5 * ((R - G) + (R - B))
#     den = np.sqrt((R - G)**2 + (R - B) * (G - B))
#     theta = np.arccos(num / (den + 1e-6))  # Avoid division by zero
    
#     H = np.copy(B)  # Initialize H array
#     H[B <= G] = theta[B <= G]
#     H[B > G] = 2 * np.pi - theta[B > G]
#     H = H / (2 * np.pi)  # Normalize to [0, 1]
    
#     # Merge H, S, and I channels
#     hsi_image = cv2.merge([H, S, I])
    
#     return hsi_image


# def bgr_to_luma(image):
#     image = image.astype(np.float32) / 255.0
#     B, G, R = cv2.split(image)
#     # Calculate luma based on the Rec. 601 standard
#     luma = 0.299 * R + 0.587 * G + 0.114 * B
#     # Scale back to [0, 255]
#     # luma = (luma * 255).astype(np.uint8)
    
#     return luma


# def Haar_wavelet(img):
#     H, W = img.shape[:2]

#     # divisible to 2
#     H = int(H & ~1)
#     W = int(W & ~1)

#     horizontal = np.zeros((H,W), dtype=np.float64)
#     out = np.zeros((H, W), dtype=np.float64)

#     m = (img[:H, :W:2] + img[:H, 1:W:2])/2
#     horizontal[:, :W//2] = m
#     horizontal[:, W//2:] = img[:H, :W:2] - m

#     m = (horizontal[:H:2, :] + horizontal[1:H:2, :]) / 2
#     out[:H//2, :] = m
#     out[H//2:, :] = horizontal[:H:2, :] - m

#     # for i in range(H):
#     #     for j in range(int(W/2)):
#     #         m = (img[i, 2*j] + img[i, 2*j+1]) / 2
#     #         horizontal[i, j] = m
#     #         horizontal[i, j + int(W/2)] = img[i, 2*j] - m
    
#     # for i in range(int(H/2)):
#     #     for j in range(W):
#     #         m = (horizontal[2*i, j] + horizontal[2*i+1, j]) / 2
#     #         out[i, j] = m
#     #         out[i + H/2, j] = horizontal[2*i, j] - m

#     LL = out[: H//2, : W//2]
#     HL = out[: H//2, W//2 :]
#     LH = out[H//2 :, : W//2]
#     HH = out[H//2 :, W//2 :]

#     return LL, (LH, HL, HH) 

##################################################################################
# percentiles = [1, 5, 10, 15, 90, 95, 99]

def calc_laplacian_var(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_value = cv2.Laplacian(img, cv2.CV_16S, ksize=3).var()

    return lap_value

def calc_brightness(img):
    image = img.copy()
    if len(image.shape) == 3:
        b, g, r = (
            image[:, :, 0].astype("int"),
            image[:, :, 1].astype("int"),
            image[:, :, 2].astype("int"),
        )
        pixel_brightness = (
        np.sqrt(0.241 * (r * r) + 0.691 * (g * g) + 0.068 * (b * b))
    ) / 255
    else:
        pixel_brightness = image / 255.0

    percentiles = [5, 99]
    perc_values = np.percentile(pixel_brightness, percentiles)

    brightness = 1 - perc_values[0] # percentiles_5
    darkness = perc_values[1] # percentiles_99

    return brightness, darkness


def calc_entropy(img) -> float:
    image = Image.fromarray(img[..., ::-1])
    entropy = image.entropy()
    assert isinstance(
        entropy, float
    )  # PIL does not have type ann stub so need to assert function return
    return entropy


def calc_blurriness(img):
    image = Image.fromarray(img[..., ::-1])
    gray_image = image.convert('L')
    edges = gray_image.filter(ImageFilter.FIND_EDGES)
    blurriness = ImageStat.Stat(edges).var[0]
    return np.sqrt(blurriness)  # type:ignore

    

    






