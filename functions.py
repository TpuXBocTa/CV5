import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


def Blur(image, size=5, sigma=3, showcase=False):
    blurred = cv2.GaussianBlur(image, (size, size), sigma)

    if showcase:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(blurred)
        plt.title("Gaussian Blur")
        plt.axis("off")
        plt.show()

    return blurred


def KMeans(image, k=4, attempts=10, showcase=False):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    ret, label, center = cv2.kmeans(
        Z,
        k,
        None,
        criteria,
        attempts,
        cv2.KMEANS_RANDOM_CENTERS
    )

    center = np.uint8(center)
    res = center[label.flatten()]
    clustered_image = res.reshape((image.shape))

    if showcase:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(clustered_image)
        plt.title(f"K-Means (k={k})")
        plt.axis("off")
        plt.show()

    return clustered_image


def EnhanceContrast(image, clip_limit=2.0, tile_grid_size=(8, 8), showcase=False):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    if showcase:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(enhanced)
        plt.title("Enhanced Contrast")
        plt.axis("off")
        plt.show()

    return enhanced


def RemoveColor(image, target_hsv=(0, 255, 255), hue_tolerance=25, sv_tolerance=200, showcase=False):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
    h, s, v = cv2.split(hsv)
    H0, S0, V0 = target_hsv

    hue_diff = np.minimum(np.abs(h - H0), 180 - np.abs(h - H0))

    dist = np.sqrt(
        (hue_diff / hue_tolerance) ** 2 +
        ((s - S0) / sv_tolerance) ** 2 +
        ((v - V0) / sv_tolerance) ** 2
    )

    mask = np.uint8(dist < 1) * 255
    mask_inv = cv2.bitwise_not(mask)

    result = cv2.bitwise_and(image, image, mask=mask_inv)

    if showcase:
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(result)
        plt.title("Color Removed")
        plt.axis("off")

        plt.show()

    return result


def CountAreasInPolygon(process_image, display_image, points, min_area=3, save_dir=".", showcase=False):
    if len(process_image.shape) == 3:
        gray = cv2.cvtColor(process_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = process_image.copy()

    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = np.zeros_like(binary)
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)

    masked = cv2.bitwise_and(binary, mask)

    contours, _ = cv2.findContours(
        masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    buildings = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    count = len(buildings)

    vis_proc = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    vis_disp = display_image.copy()

    cv2.polylines(vis_proc, [pts], True, (0, 255, 0), 1)
    cv2.polylines(vis_disp, [pts], True, (0, 255, 0), 1)

    cv2.drawContours(vis_proc, buildings, -1, (0, 0, 255), 1)
    cv2.drawContours(vis_disp, buildings, -1, (0, 0, 255), 1)

    if showcase:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(vis_proc)
        plt.title(f"Areas found: {count}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(vis_disp)
        plt.title("Result Visualization")
        plt.axis("off")
        plt.show()

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path_proc = os.path.join(save_dir, f"processed_{timestamp}.png")
    save_path_disp = os.path.join(save_dir, f"visualized_{timestamp}.png")

    cv2.imwrite(save_path_proc, cv2.cvtColor(vis_proc, cv2.COLOR_RGB2BGR))
    cv2.imwrite(save_path_disp, cv2.cvtColor(vis_disp, cv2.COLOR_RGB2BGR))

    return count, vis_proc, vis_disp


def IncreaseSaturation(image, saturation_scale=1.5, showcase=False):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    s *= saturation_scale
    s = np.clip(s, 0, 255)

    hsv_enhanced = cv2.merge((h, s, v)).astype(np.uint8)
    enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

    if showcase:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(enhanced)
        plt.title("Increased Saturation")
        plt.axis("off")
        plt.show()

    return enhanced


def EdgeDetect(image, low_threshold=50, high_threshold=150, aperture_size=3, use_L2gradient=True, showcase=False):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (3, 3), 1.0)

    edges = cv2.Canny(
        blurred,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=aperture_size,
        L2gradient=use_L2gradient
    )

    if showcase:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        if len(image.shape) == 2:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(edges, cmap="gray")
        plt.title(f"Edges (Canny {low_threshold}/{high_threshold})")
        plt.axis("off")

        plt.show()

    return edges


def HighlightKMeansClusterByTargetColor(original_image, kmeans_image, target_color=(32, 92, 24), color_tolerance=80.0,
                                        alpha=0.25, save_dir=".", name="", showcase=False, noise_min=30.0,
                                        noise_max=300.0, secondary_color_tolerance=80.0, border_erosion=2, ):
    if original_image.shape[:2] != kmeans_image.shape[:2]:
        kmeans_image = cv2.resize(
            kmeans_image,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    pixels = kmeans_image.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)

    brightness = (
            0.299 * unique_colors[:, 0] +
            0.587 * unique_colors[:, 1] +
            0.114 * unique_colors[:, 2]
    )

    sorted_indices = np.argsort(brightness)

    target_color_arr = np.array(target_color, dtype=np.float32)

    chosen_color = None
    second_color = None

    first_dist = None
    second_dist = None
    noise_level = None
    second_used = False

    for idx in sorted_indices:
        color = unique_colors[idx].astype(np.float32)
        dist = np.linalg.norm(color - target_color_arr)
        if dist <= color_tolerance:
            chosen_color = unique_colors[idx]
            first_dist = float(dist)
            break

    if chosen_color is None:
        chosen_color = unique_colors[sorted_indices[0]]
        first_dist = float(
            np.linalg.norm(chosen_color.astype(np.float32) - target_color_arr)
        )

    chosen_color_arr = chosen_color.astype(np.float32)
    for idx in sorted_indices:
        color = unique_colors[idx]
        if np.array_equal(color, chosen_color):
            continue
        color_f = color.astype(np.float32)
        dist2 = np.linalg.norm(color_f - chosen_color_arr)
        if dist2 <= secondary_color_tolerance:
            second_color = color
            second_dist = float(dist2)
            break

    color_mask = np.all(
        kmeans_image == chosen_color.reshape(1, 1, 3),
        axis=2
    ).astype(np.uint8) * 255

    if second_color is not None:
        second_mask = np.all(
            kmeans_image == second_color.reshape(1, 1, 3),
            axis=2
        ).astype(np.uint8)

        if np.any(second_mask):
            gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

            second_mask_u8 = (second_mask * 255).astype(np.uint8)

            if border_erosion > 0:
                kernel = np.ones((3, 3), np.uint8)
                inner_mask_u8 = cv2.erode(
                    second_mask_u8,
                    kernel,
                    iterations=border_erosion
                )
            else:
                inner_mask_u8 = second_mask_u8

            if not np.any(inner_mask_u8):
                inner_mask_u8 = second_mask_u8

            inner_mask_bool = inner_mask_u8.astype(bool)

            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_values = lap[inner_mask_bool]

            noise_level = float(lap_values.var()) if lap_values.size > 0 else 0.0

            if noise_min <= noise_level <= noise_max:
                color_mask_second = (second_mask * 255).astype(np.uint8)
                color_mask = cv2.bitwise_or(color_mask, color_mask_second)
                second_used = True
            else:
                second_used = False

    print("=== HighlightKMeansClusterByTargetColor DEBUG ===")
    print(f"Target color: {target_color}")
    print(f"Chosen 1st cluster color: {tuple(int(x) for x in chosen_color)}")
    print(f"Distance 1st -> target: {first_dist:.2f} (tolerance={color_tolerance})")
    if second_color is not None:
        print(f"2nd candidate color: {tuple(int(x) for x in second_color)}")
        print(
            f"Distance 2nd -> 1st: {second_dist:.2f} "
            f"(tolerance={secondary_color_tolerance})"
        )
        print(
            f"Noise level (inner region) in 2nd: {noise_level:.2f} "
            f"(range=[{noise_min}, {noise_max}]), used={second_used}, "
            f"border_erosion={border_erosion}"
        )
    else:
        print("No 2nd candidate color found within secondary tolerance.")
    print("===============================================")

    highlighted = original_image.copy()
    highlight_color = np.array([255, 255, 255], dtype=np.float32)  # синій

    mask_bool = (color_mask == 255)

    highlighted_float = highlighted.astype(np.float32)
    highlighted_float[mask_bool] = (
            (1.0 - alpha) * highlighted_float[mask_bool] +
            alpha * highlight_color
    )
    highlighted = np.clip(highlighted_float, 0, 255).astype(np.uint8)

    if showcase:
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(kmeans_image)
        plt.title("K-Means Image")
        plt.axis("off")

        title_lines = [
            f"1st: {tuple(int(x) for x in chosen_color)}",
            f"dist1={first_dist:.1f}, tol1={color_tolerance}",
        ]

        if second_color is not None:
            title_lines.append(
                f"2nd: {tuple(int(x) for x in second_color)}, "
                f"dist2={second_dist:.1f}, tol2={secondary_color_tolerance}"
            )
            if noise_level is not None:
                title_lines.append(
                    f"noise={noise_level:.1f}, "
                    f"range=[{noise_min}, {noise_max}], used={second_used}"
                )
                title_lines.append(f"border_erosion={border_erosion}")

        title_text = "\n".join(title_lines)

        plt.subplot(1, 3, 3)
        plt.imshow(highlighted)
        plt.title(title_text)
        plt.axis("off")

        plt.show()

    os.makedirs(save_dir, exist_ok=True)

    if not name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"highlighted_cluster_{timestamp}.png"

    save_path = os.path.join(save_dir, name)
    cv2.imwrite(save_path, cv2.cvtColor(highlighted, cv2.COLOR_RGB2BGR))

    return highlighted, tuple(int(x) for x in chosen_color)
