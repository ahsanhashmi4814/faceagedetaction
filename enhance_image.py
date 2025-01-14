import cv2
import numpy as np

def enhance_image(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is None:
        print("Error: Unable to load image.")
        return

    # Resize the image (optional)
    image = cv2.resize(image, (800, 1100))

    # 1. Adjust brightness and contrast
    alpha = 1.0  # Contrast control (1.0-3.0)
    beta = 12    # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 2. Apply sharpening filter
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(adjusted, -1, kernel)

    # 3. Denoise the image (optional)
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 8, 10, 7, 21)

    # Save the enhanced image
    cv2.imwrite(output_path, denoised)
    print(f"Enhanced image saved to {output_path}")

    # Display the original and enhanced images
    cv2.imshow('Original Image', image)
    cv2.imshow('Enhanced Image', denoised)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Enhance the image
image_path = 'input_image.png'  # Replace with your image file path
output_path = 'enhanced_image.jpg'
enhance_image(image_path, output_path)
