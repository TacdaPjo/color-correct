## Author - Johan Elfing ##
# import warnings

# warnings.filterwarnings('ignore')

import numpy as np
import os
import colour.utilities

from colour_checker_detection import (detect_colour_checkers_segmentation, SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)


# image_directory = 'C:/Users/Johan/color-correct/Pictures3'
# image_pattern = 'IMG_8177.png'
# image_path = os.path.join(image_directory, image_pattern)


# image_path = self_url (self_url = path)
def process_url(self_url: str):
    # Example function body
    print(f"Pawel Pawelisnki: {self_url}")

    image = colour.cctf_decoding(colour.io.read_image(self_url))
    #
    # # colour.plotting.plot_image(colour.cctf_encoding(image), title='Original Image')
    #
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.update(
        {
            "aspect_ratio_minimum": 1.2,  # Adjust as needed
            "aspect_ratio_maximum": 2.0,  # Adjust as needed
            "swatches_count_minimum": 18,  # Adjust as needed
            "swatches_count_maximum": 24,  # Adjust as needed
            "swatch_minimum_area_factor": 150,  # Adjust as needed
            "swatch_contour_scale": 1.5,  # Adjust as needed
        }
    )
    print(f"Processing URL 2: {self_url}")
    #
    # print("Updated Settings for ColorChecker Classic:")
    # for key, value in SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.items():
    #    print(f"{key}: {value}")

    SWATCHES = []
    colour_checker_data = detect_colour_checkers_segmentation(
        image, additional_data=True, **SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)

    if colour_checker_data:
        for data in colour_checker_data:
            swatch_colours, swatch_masks, colour_checker_image = (data.values)
            SWATCHES.append(swatch_colours)

            masks_i = np.zeros(colour_checker_image.shape)
            for i, mask in enumerate(swatch_masks):
                masks_i[mask[0]:mask[1], mask[2]:mask[3], ...] = 1

            # colour.plotting.plot_image(colour.cctf_encoding(np.clip(colour_checker_image + masks_i * 0.25, 0, 1)))

    D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    REFERENCE_COLOUR_CHECKER = colour.CCS_COLOURCHECKERS[
        'ColorChecker24 - After November 2014']
    print(f"Processing URL 3: {self_url}")
    colour_checker_rows = REFERENCE_COLOUR_CHECKER.rows
    colour_checker_columns = REFERENCE_COLOUR_CHECKER.columns

    REFERENCE_SWATCHES = colour.XYZ_to_RGB(
        colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
        'sRGB', REFERENCE_COLOUR_CHECKER.illuminant)

    for i, swatches in enumerate(SWATCHES):
        swatches_xyY = colour.XYZ_to_xyY(colour.RGB_to_XYZ(
            swatches, 'sRGB', D65))

        colour_checker = colour.characterisation.ColourChecker(
            os.path.basename(self_url),
            dict(zip(REFERENCE_COLOUR_CHECKER.data.keys(), swatches_xyY)),
            D65, colour_checker_rows, colour_checker_columns)

        # colour.plotting.plot_multi_colour_checkers(
        #    [REFERENCE_COLOUR_CHECKER, colour_checker])
        print(f"Processing URL 4: {self_url}")

        swatches_f = colour.colour_correction(swatches, swatches, REFERENCE_SWATCHES)
        swatches_f_xyY = colour.XYZ_to_xyY(colour.RGB_to_XYZ(
            swatches_f, 'sRGB', D65))
        colour_checker = colour.characterisation.ColourChecker(
            '{0} - CC'.format(os.path.basename(self_url)),
            dict(zip(REFERENCE_COLOUR_CHECKER.data.keys(), swatches_f_xyY)),
            D65, colour_checker_rows, colour_checker_columns)

        # colour.plotting.plot_multi_colour_checkers(
        #   [REFERENCE_COLOUR_CHECKER, colour_checker])
        print(f"Processing URL 5: {self_url}")

        colour.plotting.plot_image(colour.cctf_encoding(
            colour.colour_correction(
                image, swatches, REFERENCE_SWATCHES)))

    return
