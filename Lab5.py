import cv2
import functions

images = 10
for i in range(images):
    img_bgr = cv2.imread(str(i) + ".png")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    forest_blured = functions.Blur(img_rgb, 15, 15)
    forest_kmeaned = functions.KMeans(forest_blured, 4, 5)
    forest_mask = functions.HighlightKMeansClusterByTargetColor(img_rgb, forest_kmeaned, name=str(i) + "highlited.png",
                                                                noise_min=2.5, noise_max=5,
                                                                secondary_color_tolerance=30)
    forest_removed = functions.ApplyMaskAsBlack(img_rgb, forest_mask)

    fields_blured = functions.Blur(forest_removed, 15, 15)
    fields_kmeaned = functions.KMeans(fields_blured, 5, 5)
    fields_merged_means = functions.MergeSimilarKMeansClusters(fields_kmeaned, color_tolerance=40)
    edge_mask = functions.EdgeDetect(img_rgb, merge_kernel_size=20)
    fields_meaned_edged = functions.ApplyMaskAsBlack(fields_merged_means, edge_mask)
    field_mask = functions.HighlightKMeansRegionsExcludeColors(
        img_rgb,
        fields_meaned_edged, target_colors=[(190, 190, 190), (0, 0, 0)], color_tolerance=60, min_area=3000,
        max_hole_ratio=0.1, rect_fill_min=0.45, showcase=False)
    result = functions.HighlightZonesWithMasks(img_rgb, [forest_mask, field_mask], [(0, 255, 0), (255, 0, 0)],
                                               labels=["Forest", "Fields"], alpha=0.25, showcase=True,
                                               name=str(i) + "zones.png")
