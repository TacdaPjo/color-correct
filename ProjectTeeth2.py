## Author - Johan Elfing ##
# import warnings
#
# warnings.filterwarnings('ignore')

import numpy as np
import os
import colour.utilities
from PIL import Image as im
from colour_checker_detection import (detect_colour_checkers_segmentation, SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)
import gc

image_directory = 'C:/Users/Johan/color-correct/Pictures3/realTest'
image_pattern = 't9.png'
image_path = os.path.join(image_directory, image_pattern)

original_image_pil = im.open(image_path)

image_mode = original_image_pil.mode

# Determine bit depth per channel
if image_mode in ['1', 'L', 'P']:
    bits_per_channel = 8
elif image_mode == 'RGB':
    bits_per_channel = 8 * 3
elif image_mode == 'RGBA':
    bits_per_channel = 8 * 4
else:
    bits_per_channel = 8 * len(image_mode)

total_bit_depth = bits_per_channel

# Determine if the image is 24-bit or 32-bit
if total_bit_depth == 24:
    print("The image is 24-bit.")
elif total_bit_depth == 32:
    print("The image is 32-bit.")
else:
    print(f"The image has a different bit depth: {total_bit_depth}-bit.")

resized_dimensions = (1024, 680)
original_image_pil = original_image_pil.resize(resized_dimensions)

normalized_image = np.asarray(original_image_pil,
                              dtype=np.float32) / 255.0  # Normalize the image and use float32 to save memory
processed_image = colour.cctf_decoding(colour.io.read_image(image_path))

colour.plotting.plot_image(colour.cctf_encoding(normalized_image), title='normalized_image')
colour.plotting.plot_image(colour.cctf_encoding(processed_image), title='processed_image')
new_dimensions = (6264, 4180)

original_width, original_height = normalized_image.shape[1], normalized_image.shape[
    0]  # Use shape attribute to get width and height

# Calculate scaling factors for width and height
scaling_factor_width = original_width / new_dimensions[0]
scaling_factor_height = original_height / new_dimensions[1]

# Use the average scaling factor for uniform scaling
scaling_factor = (scaling_factor_width + scaling_factor_height) / 2

# print("ScalingFactorWidth:", 150 * scaling_factor)

SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.update(
    {
        "aspect_ratio_minimum": 1.2,  # No change needed
        "aspect_ratio_maximum": 2.0,  # No change needed
        "swatches_count_minimum": 18,  # No change needed
        "swatches_count_maximum": 24,  # No change needed
        "swatch_minimum_area_factor": 150 * scaling_factor,  # Adjusted using scaling factor for area
        "swatch_contour_scale": 1.5,  # Adjusted using scaling factor for width
    }
)

SWATCHES = []
colour_checker_data = detect_colour_checkers_segmentation(
    processed_image, additional_data=True, **SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)

if colour_checker_data:
    for data in colour_checker_data:
        swatch_colours, swatch_masks, colour_checker_image = (data.values)
        SWATCHES.append(swatch_colours)

        masks_i = np.zeros(colour_checker_image.shape, dtype=np.float32)
        for i, mask in enumerate(swatch_masks):
            masks_i[mask[0]:mask[1], mask[2]:mask[3], ...] = 1

REFERENCE_COLOUR_CHECKER = colour.CCS_COLOURCHECKERS[
    'ColorChecker24 - After November 2014']

REFERENCE_SWATCHES = colour.XYZ_to_RGB(
    colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
    'sRGB', REFERENCE_COLOUR_CHECKER.illuminant)

gamma_corrected_image = None

for i, swatches in enumerate(SWATCHES):
    colour.plotting.plot_image(
        colour.cctf_encoding(colour.colour_correction(processed_image, swatches, REFERENCE_SWATCHES)))
    gamma_corrected_image = colour.cctf_encoding(
        colour.colour_correction(processed_image, swatches, REFERENCE_SWATCHES))

# Ensure the image is in the correct range and type for saving
gamma_corrected_image = np.clip(gamma_corrected_image, 0, 1)  # Clip values to ensure they are in the range [0, 1]
gamma_corrected_image = (gamma_corrected_image * 255).astype(np.uint8)  # Convert to uint8

corrected_image_path = os.path.join(image_directory, 'corrected_imaget9.png')
corrected_image_pil = im.fromarray(gamma_corrected_image)
corrected_image_pil.save(corrected_image_path)

# colour.plotting.plot_image(corrected_image, title='Original Image')

# corrected_image_path = os.path.join(image_directory, 'corrected_imaget7.png')
# corrected_image_pil = im.fromarray((corrected_image * 255).astype(np.uint8))
# corrected_image_pil.save(corrected_image_path)
