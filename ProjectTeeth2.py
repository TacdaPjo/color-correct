import numpy as np
import os
from PIL import Image as im
import colour
from colour_checker_detection import detect_colour_checkers_segmentation, SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC
import matplotlib.pyplot as plt
import rawpy
import imageio

ILLUMINANT_D65 = colour.CCS_ILLUMINANTS['cie_2_1931']['D65']

RGB_COLOURSPACE = colour.RGB_COLOURSPACES['sRGB']


def convert_single_raw_to_png(input_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.png')
    with rawpy.imread(input_file) as raw:
        rgb = raw.postprocess()
        imageio.imsave(output_file, rgb, format='png')

    print(f'File converted and saved as {output_file}.')


def convert_swatch_colours_to_rgb(swatch_colours: np.ndarray) -> np.ndarray:
    # Assumes swatch_colours is gamma-corrected sRGB
    swatch_colours = np.clip(swatch_colours, 0, 1)

    # No gamma correction needed if input is already gamma-corrected
    rgb_colours = np.clip(swatch_colours * 255, 0, 255).astype(int)
    return rgb_colours


def rgb_to_lab(rgb):
    # Convert RGB to XYZ
    rgb = np.array(rgb, dtype=np.float64)

    if np.max(rgb) > 1.0:
        # If RGB values are in the range 0-255, normalize them
        rgb = rgb / 255.0
    xyz = colour.RGB_to_XYZ(rgb, RGB_COLOURSPACE.whitepoint, RGB_COLOURSPACE.whitepoint,
                            RGB_COLOURSPACE.matrix_RGB_to_XYZ)

    # Convert XYZ to LAB
    lab = colour.XYZ_to_Lab(xyz, illuminant=ILLUMINANT_D65)

    return lab


# Function to plot the camera vs reference RGB values
def plot_rgb_comparison(camera_rgb, reference_rgb, title="Camera vs Reference RGB Values"):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    channels = ['Red', 'Green', 'Blue']

    for i in range(3):
        ax[i].scatter(range(len(camera_rgb)), camera_rgb[:, i], color='red', label='Camera')
        ax[i].scatter(range(len(reference_rgb)), reference_rgb[:, i], color='blue', label='Reference')

        # Annotate each point with the swatch index
        for j in range(len(camera_rgb)):
            ax[i].annotate(f'{j}', (j, camera_rgb[j, i]), textcoords="offset points", xytext=(0, 5), ha='center',
                           color='red', fontsize=9)
            ax[i].annotate(f'{j}', (j, reference_rgb[j, i]), textcoords="offset points", xytext=(0, 5), ha='center',
                           color='blue', fontsize=9)

        ax[i].set_title(f"{channels[i]} Channel")
        ax[i].legend()
        ax[i].set_xlabel("Swatch Index")
        ax[i].set_ylabel("Value (0-255)")

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Function to calculate Delta E
def delta_e(lab1, lab2):
    return np.sqrt(np.sum((lab1 - lab2) ** 2))


def delta_e_rgb(rgb1, rgb2):
    rgb1_array = np.array(rgb1)
    rgb2_array = np.array(rgb2)
    return np.sqrt(np.sum((np.array(rgb1_array) - np.array(rgb2_array)) ** 2))


# Custom RGB values
ref_rgb_values = {
    'Dark Skin': [115, 81, 68],
    'Light Skin': [194, 150, 130],
    'Blue Sky': [98, 122, 157],
    'Foliage': [87, 108, 67],
    'Blue Flower': [133, 128, 177],
    'Bluish Green': [103, 189, 170],
    'Orange': [214, 126, 44],
    'Purple Red': [80, 91, 166],
    'Moderate Red': [193, 90, 99],
    'Purple': [94, 60, 108],
    'Yellow Green': [157, 188, 64],
    'Orange Yellow': [224, 163, 46],
    'Blue': [56, 61, 150],
    'Green': [70, 148, 73],
    'Red': [175, 54, 60],
    'Yellow': [231, 199, 31],
    'Magenta': [187, 86, 149],
    'Cyan': [8, 133, 161],
    'White': [243, 243, 242],
    'Neutral 8': [200, 200, 200],
    'Neutral 65': [160, 160, 160],
    'Neutral 5': [122, 122, 121],
    'Neutral 35': [85, 85, 85],
    'Black': [52, 52, 52]
}

# Reference LAB values
lab_values = [
    [37.99, 13.56, 14.06],  # Dark Skin
    [65.71, 18.13, 17.81],  # Light Skin
    [49.93, -4.88, -21.93],  # Blue Sky
    [43.14, -13.10, 21.91],  # Foliage
    [55.11, 8.84, -25.40],  # Blue Flower
    [70.72, -33.40, -0.19],  # Bluish Green
    [62.66, 36.07, 57.10],  # Orange
    [40.02, 10.41, -45.96],  # Purplish Blue
    [51.12, 48.24, 16.25],  # Moderate Red
    [30.33, 22.98, -21.59],  # Purple
    [72.53, -23.71, 57.26],  # Yellow Green
    [71.94, 19.36, 67.86],  # Orange Yellow
    [28.78, 14.18, -50.30],  # Blue
    [55.26, -38.34, 31.37],  # Green
    [42.10, 53.38, 28.19],  # Red
    [81.73, 4.04, 79.82],  # Yellow
    [51.94, 49.99, -14.57],  # Magenta
    [51.04, -28.63, -28.64],  # Cyan
    [96.53, -0.48, 1.89],  # White
    [81.26, -0.64, 0.44],  # Neutral 8
    [66.77, -0.73, 0.48],  # Neutral 6.5
    [50.87, -0.15, 0.23],  # Neutral 5
    [35.66, -0.42, -0.14],  # Neutral 3.5
    [20.46, -0.08, -0.97]  # Black
]

# Normalize custom RGB values
custom_rgb_values_normalized = {k: np.array(v) / 255.0 for k, v in ref_rgb_values.items()}

reference_colour_checker = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']

reference_lab_values = reference_colour_checker.data.values()

reference_rgb_values = np.array(list(ref_rgb_values.values()))  # Convert dictionary to array

REFERENCE_SWATCHES = colour.XYZ_to_RGB(
    colour.xyY_to_XYZ(list(reference_colour_checker.data.values())),
    'sRGB', reference_colour_checker.illuminant)

RGB_COLOURSPACE = colour.RGB_COLOURSPACES['sRGB']

XYZ_values = colour.Lab_to_XYZ(lab_values, illuminant=RGB_COLOURSPACE.whitepoint)

RGB_values_linear = colour.XYZ_to_RGB(
    XYZ_values,
    colourspace=RGB_COLOURSPACE,  # Specify the colourspace
    illuminant=RGB_COLOURSPACE.whitepoint,
    matrix=RGB_COLOURSPACE.matrix_XYZ_to_RGB,
    chromatic_adaptation_transform=None,  # Use if no adaptation is needed
    apply_cctf_decoding=False  # Skip gamma correction
)
# Example usage
# C:/Users/Johan/color-correct/Pictures3/newImages/Unproccess
# C:/Users/Johan/color-correct/Pictures3/newImages/Proccessed
# C:/Users/Johan/color-correct/Pictures3/newImages/Calibrated
# 'C:/Users/Johan/color-correct/Pictures3/newImages/bilder0829/raw/ringinlangt.CR3'
# 'C:/Users/Johan/color-correct/Pictures3/newImages/bilder0829/proc'
# 'C:/Users/Johan/color-correct/Pictures3/newImages/bilder0829/Cali'

# Lösa så att man kan både använda CR3 & DNG.
name = 'IMG_8823'
input_dir = f'C:/Users/Johan/color-correct/Pictures3/newImages/bilder0901/Unprocess/{name}.CR3'
output_dir = 'C:/Users/Johan/color-correct/Pictures3/newImages/bilder0901/Process'
calibrated_dir = 'C:/Users/Johan/color-correct/Pictures3/newImages/bilder0901/Calibrated'
convert_single_raw_to_png(input_dir, output_dir)
# Load and process the image
image_directory = output_dir
# Normal Image
# image_path = os.path.join(image_directory, 't2.png')
# Dark Image
image_path = os.path.join(image_directory, f'{name}.png')

original_image_pil = im.open(image_path)
original_image_pil = original_image_pil.resize((1024, 680))
normalized_image = np.asarray(original_image_pil, dtype=np.float32) / 255.0
#processed_image = colour.cctf_decoding(normalized_image, function='sRGB')

new_dimensions = (6264, 4180)
original_width, original_height = normalized_image.shape[1], normalized_image.shape[0]
scaling_factor_width = original_width / new_dimensions[0]
scaling_factor_height = original_height / new_dimensions[1]
scaling_factor = (scaling_factor_width + scaling_factor_height) / 2

# Adjust color checker detection settings
SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.update({
    "aspect_ratio_minimum": 1.2,
    "aspect_ratio_maximum": 2.0,
    "swatches_count_minimum": 18,
    "swatches_count_maximum": 24,
    "swatch_minimum_area_factor": 250 * scaling_factor,
    "swatch_contour_scale": 1.5,
})

# Detect color checkers
SWATCHES = []
colour_checker_data = detect_colour_checkers_segmentation(
    normalized_image, additional_data=True, **SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)

if colour_checker_data:
    for data in colour_checker_data:
        swatch_colours, swatch_masks, colour_checker_image = data.values

        swatch_colours_rgb = []
        for swatch in swatch_colours:
            try:
                RGB_val = convert_swatch_colours_to_rgb(swatch)
                print("RGB_VAL", RGB_val)
                swatch_colours_rgb.append(RGB_val)
            except Exception as e:
                print(f"Error processing swatch: {e}")

        SWATCHES.append(swatch_colours_rgb)

perceptibility_threshold = 3.3
acceptability_threshold = 7

# Start calibration process

for i, swatches in enumerate(SWATCHES):
    camera_rgb_values = np.array(swatches)
    reference_rgb_values = ref_rgb_values

    if len(swatches) != len(ref_rgb_values):
        print(
            f"Warning: Number of detected swatches ({len(swatches)}) does not match the number of custom RGB values ({len(ref_rgb_values)}).")
        continue

    reference_rgb_values_array = np.array(list(ref_rgb_values.values()), dtype=np.float64)

    camera_rgb_values_array = np.array(camera_rgb_values, dtype=np.float64)

    if np.max(camera_rgb_values_array) <= 1.0:
        camera_rgb_values_array = camera_rgb_values_array * 255.0

    camera_rgb_values_array = np.clip(camera_rgb_values_array, 0, 255).astype(np.uint8)

    # /Bilder0829/IMG_8828 kalibrerad.png'
    # CalibratedImageImg_8879
    # image_path2 = 'C:/Users/Johan/color-correct/Pictures3/newImages/CalibratedImageImg_8879.png'
    # original_image_pil2 = im.open(image_path2)
    # original_image_pil2 = original_image_pil2.resize((1024, 680))
    # normalized_image2 = np.asarray(original_image_pil2, dtype=np.float32) / 255.0

    # Apply colour correction
    corrected_image = colour.colour_correction(normalized_image, camera_rgb_values_array, reference_rgb_values_array)

    # Convert corrected image for display
    corrected_image_to_show = np.clip(corrected_image, 0, 1)  # Ensure it's in [0,1]
    corrected_image_to_show = (corrected_image_to_show * 255).astype(np.uint8)
    corrected_image_pil = im.fromarray(corrected_image_to_show)

    # Display corrected image using PIL
    corrected_image_pil.show(title="Corrected Image")

    # Detect color checkers again on the corrected image
    corrected_swatches = []
    corrected_image_checker_data = detect_colour_checkers_segmentation(
        corrected_image, additional_data=True, **SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)

    if corrected_image_checker_data:
        for data in corrected_image_checker_data:
            swatch_colours, swatch_masks, colour_checker_image = data.values
            corrected_swatches.append(swatch_colours)
    else:
        print("Warning: No color checkers detected after calibration.")
        corrected_swatches = SWATCHES  # Fallback to original swatches if none detected

    total_delta_e = 0
    num_swatches = len(corrected_swatches[0]) if corrected_swatches else 0

    if num_swatches > 0:
        for j, swatch in enumerate(corrected_swatches[0]):
            swatch_lab = rgb_to_lab(swatch)
            RGB_val2 = convert_swatch_colours_to_rgb(swatch)

            ref_swatch_name = list(ref_rgb_values.keys())[j]
            reference_rgb = ref_rgb_values[ref_swatch_name]
            reference_lab = rgb_to_lab(reference_rgb)

            delta_e_value = delta_e(swatch_lab, reference_lab)

            total_delta_e += delta_e_value

            perceptibility = delta_e_value < perceptibility_threshold
            acceptability = delta_e_value < acceptability_threshold

            if perceptibility:
                print(f"Swatch {j + 1}: Imperceptible color difference (Delta E = {delta_e_value})")
            elif acceptability:
                print(f"Swatch {j + 1}: Acceptable color difference (Delta E = {delta_e_value})")
            else:
                print(f"Swatch {j + 1}: Unacceptable color difference (Delta E = {delta_e_value})")
        # Average_delta_e skall skickas -->
        average_delta_e = total_delta_e / num_swatches
        print(f"\nAverage Delta E After Calibration: {average_delta_e}")
    else:
        print("No swatches available for Delta E calculation after calibration.")
corrected_image_path = os.path.join(calibrated_dir, f'{name}.png')
corrected_image_pil = im.fromarray(corrected_image_to_show)
corrected_image_pil.save(corrected_image_path)

reference_rgb_values_array = np.array(list(ref_rgb_values.values()), dtype=np.uint8)
camera_rgb_values = np.array(camera_rgb_values, dtype=np.uint8)

# Plot comparison
plot_rgb_comparison(camera_rgb_values, reference_rgb_values_array)

######################################################################################################
# image_mode = original_image_pil.mode
#
# if image_mode in ['1', 'L', 'P']:
#     bits_per_channel = 8
# elif image_mode == 'RGB':
#     bits_per_channel = 8 * 3
# elif image_mode == 'RGBA':
#     bits_per_channel = 8 * 4
# else:
#     bits_per_channel = 8 * len(image_mode)
#
# total_bit_depth = bits_per_channel
#
# if total_bit_depth == 24:
#     print("The image is 24-bit.")
# elif total_bit_depth == 32:
#     print("The image is 32-bit.")
# else:
#     print(f"The image has a different bit depth: {total_bit_depth}-bit.")