import sys
import os
sys.path.append(".")
# Add the parent directory to sys.path so utils can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import cv2
import matplotlib.pyplot as plt
from utils import show_images, save_images, scale_down, separate_channels

if __name__ == "__main__":
    ## TODO 3.1
    ## Load Image
    ## Show it on screen
    ## Note: implement show_images in utils/functions.py
    file = 'img.png'  ## path to the image
    input_path = os.path.join("resources", "img.png")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    show_images([img], ["Display Image"])

    ## TODO 3.2
    ## Resize Image by a factor of 0.5
    ## Show it on screen
    ## Save as small.jpg
    ## Note: implement save_images, scale_down in utils/functions.py
    small_img = scale_down(img)
    show_images([small_img], ["Small Image"])
    save_images([small_img], ["small.jpg"])

    ## TODO 3.3
    ## Create and save 3 single-channel images from small image
    ## one image each channel (r, g, b)
    ## Display the channel-images on screen
    ## Note: implement separate_channels in utils/functions.py
    sep_image = separate_channels(img)
    blue, green, red = sep_image
    
    for i, img in enumerate(sep_image):
        plt.imshow(img)
        plt.axis('off')  # Ensure axis is on
        plt.savefig(f"./output/image_{i}.png", facecolor='white')  # Save with white background
        plt.close()  # Close the figure to avoid overlap 

    show_images([img, blue, green, red], ["Original", "Blue", "Green", "Red"])
    save_images([blue, green, red], ["blue.png", "green.png", "red.png"])
   
