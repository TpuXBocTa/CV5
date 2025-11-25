import cv2
import functions

images = 10
for i in range(images):
    img_bgr = cv2.imread(str(i) + ".png")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    blured = functions.Blur(img_rgb, 15, 15)
    kmeaned = functions.KMeans(blured, 4, 5)
    functions.HighlightKMeansClusterByTargetColor(img_rgb, kmeaned, name=str(i) + "highlited.png",
                                                  noise_min=2.5, noise_max=5, secondary_color_tolerance=40)
