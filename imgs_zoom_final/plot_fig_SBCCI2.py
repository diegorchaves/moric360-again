import math

import numpy as np
from PIL import Image, ImageDraw


def line_between_circles(c1, r1, c2, r2):
    """
    Returns two points (p1, p2) where the line connecting centers
    intersects the boundaries of the two circles.

    c1, c2: (x, y) centers
    r1, r2: radii
    """
    x1, y1 = c1
    x2, y2 = c2

    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)

    if dist == 0:
        return c1, c2  # degenerate case

    ux = dx / dist
    uy = dy / dist

    # point on boundary of first circle
    p1 = (x1 + ux * r1, y1 + uy * r1)

    # point on boundary of second circle
    p2 = (x2 - ux * r2, y2 - uy * r2)

    return p1, p2


def circular_zoom(
    image_path, center, roi_size, zoom_factor=2.5, bubble_center=(300, 300)
):
    """
    image_path: path to input image
    center: (x, y) center of ROI
    roi_size: size of square ROI (pixels)
    zoom_factor: how much to enlarge
    bubble_center: where to place zoomed circle
    """

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    x, y = center
    half = roi_size // 2

    # Crop ROI
    roi = img.crop((x - half, y - half, x + half, y + half))

    # Resize (zoom)
    zoom_size = int(roi_size * zoom_factor)
    roi_zoom = roi.resize((zoom_size, zoom_size), Image.NEAREST)

    # Create circular mask
    mask = Image.new("L", (zoom_size, zoom_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((0, 0, zoom_size, zoom_size), fill=255)

    # Paste zoomed ROI as circle
    bx, by = bubble_center
    img.paste(roi_zoom, (bx, by), mask)

    # Draw circle around original ROI
    draw.ellipse((x - half, y - half, x + half, y + half), outline="white", width=3)

    # Draw circle border for zoom
    draw.ellipse((bx, by, bx + zoom_size, by + zoom_size), outline="white", width=4)

    # Centers
    c1 = (x, y)
    c2 = (bx + zoom_size // 2, by + zoom_size // 2)

    # Radii
    r1 = roi_size // 2
    r2 = zoom_size // 2

    p1, p2 = line_between_circles(c1, r1, c2, r2)

    draw.line([p1, p2], fill="white", width=2)

    """
    # Draw connecting line
    draw.line(
        [(x, y), (bx + zoom_size // 2, by + zoom_size // 2)],
        fill="white",
        width=2
    )
    """

    return img


# Example usage
if __name__ == "__main__":
    output = circular_zoom(
        image_path="/home/diego/Desktop/moric360-again/imgs_zoom_final/othim19_maskerp_wsmse1_swhdc1_erp1_lambda0.0025.png",
        center=(550, 120),  # ROI center
        roi_size=120,
        zoom_factor=3,
        bubble_center=(170, 100),
    )

    output.save(
        "/home/diego/Desktop/moric360-again/imgs_zoom_final/othim19_maskerp_wsmse1_swhdc1_erp1_lambda0.0025_zoom.png"
    )
    # output.show()


"""
img = Image.open("input.jpg").convert("RGB")

rois = [
    ((400, 300), 120, (700, 50)),
    ((200, 500), 100, (700, 300)),
    ((600, 200), 80,  (700, 550)),
]

for center, size, bubble_pos in rois:
    img = circular_zoom_on_image(img, center, size, 3, bubble_pos)

img.save("multi_zoom.png")
"""
