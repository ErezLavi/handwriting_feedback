import os
from paddleocr import PaddleOCR
import cv2
from tqdm import tqdm
import random

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load random images from the media folder
folder_path = 'media'
images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Shuffle the images list to process in random order
random.shuffle(images)

for image_file in tqdm(images):
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    # Perform OCR
    results = ocr.ocr(image_path, cls=True)

    # show the image
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # check if results[0] is None
    if results[0] is None:
        print('No text detected in the image')

    else:
        # print the text detected
        print('Detected text: ')
        for word_info in results[0]:
            print(word_info[1])

        # Print the results with confidence per character
        print('\nDetected characters and their confidence score: ')
        for word_info in results[0]:
            for i in range(len(word_info[1])):
                char = word_info[1][i][0]
                conf = results[1][i]  # Get the corresponding confidence score from results[1]
                print(f"Character: {char}, Confidence: {conf}")


