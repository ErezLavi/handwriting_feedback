import os
from paddleocr import PaddleOCR
import cv2
from tqdm import tqdm
import random
from PIL import Image, ImageDraw
import numpy as np
from rapidfuzz.distance import Levenshtein
import matplotlib.pyplot as plt

# Shared folder path
my_models_folder_path = r'C:\Users\erez1\PycharmProjects\PaddleOCR_char_seg\pretrained_model\en_PP-OCRv4_rec_train'

ocr_fine_tuned_models_names = {"model_cvl_with_iam_10_epoches",
                               "model_cvl_with_iam_latest",
                               "model_cvl_with_iam_best_may_not_work",
                               "model_inference_epoch_20_iam",
                               "model_inference_epoch_30_iam",
                               "model_inference_epoch_10_cvl",
                               "model_inference_epoch_20_cvl",
                               "model_inference_latest_cvl",
                               "model_inference"}


# Function to draw bounding boxes and annotate text
def draw_results_on_image(image_path, results, title):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    if not results[0]:  # Check if no text was detected
        print(f"No text detected in {title}")
        return image

    for i, word_info in enumerate(results[0]):
        box = np.array(word_info[0]).astype(np.int32)  # Extract the bounding box
        xmin = min(box[:, 0])
        ymin = min(box[:, 1])
        xmax = max(box[:, 0])
        ymax = max(box[:, 1])

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        draw.text((xmin, ymin), f"{i}", fill="black")

    return np.array(image)


def perform_image_processing(image_path):
    img = cv2.imread(image_path, 0)
    thresh_value = 50
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # ret, thresh = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)
    # Apply adaptive thresholding for dynamic binarization
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img


def extract_actual_str(image_path):
    # Extract the original text from the image filename
    filename = os.path.basename(image_path)
    original_text = os.path.splitext(filename)[0]
    return original_text


def evaluate_image_accuracy(predicted_str, actual_str):
    if not predicted_str:
        # Predicted string is empty, treat as 0% accurate
        return 0.0
    # Calculate normalized Levenshtein distance
    normalized_distance = Levenshtein.distance(predicted_str, actual_str) / max(len(predicted_str), len(actual_str))
    # Convert distance to accuracy
    accuracy = 1 - normalized_distance
    return accuracy


def calculate_avg_model_accuracy(accuracy_scores):
    avg_accuracy_scores = {model_name: sum(scores) / len(scores) for model_name, scores in accuracy_scores.items()}
    best_model = max(avg_accuracy_scores, key=avg_accuracy_scores.get)
    return avg_accuracy_scores, best_model, avg_accuracy_scores[best_model]


def plot_accuracies_per_model(avg_accuracy_scores):
    # Extract model names and their average accuracies
    models = list(avg_accuracy_scores.keys())
    accuracies = list(avg_accuracy_scores.values())

    # Create the bar plot
    plt.figure(figsize=(10, 8))
    plt.bar(models, accuracies, color='skyblue', edgecolor='black')

    # Add title and labels
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Average Accuracy', fontsize=14)

    # Add value labels on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center', fontsize=12)

    # Rotate model names if needed
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.show()


def print_char_confidence(word_info, results):
    for i in range(len(word_info[1])):
        char = word_info[1][i][0]
        conf = results[1][i]
        print(f"Character: {char}, Confidence: {conf}")


# Initialize our fine-tuned OCR models + the original PaddleOCR model
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
models = [PaddleOCR(use_angle_cls=True, lang='en', rec_model_dir=os.path.join(my_models_folder_path, model_name))
          for model_name in ocr_fine_tuned_models_names]

# Load random images from the media folder
folder_path = 'ofir_words'
images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', 'tif'))]
random.shuffle(images)

# An array to store the accuracy scores
accuracy_scores = {model_name: [] for model_name in ocr_fine_tuned_models_names | {"PaddleOCR"}}

for image_file in tqdm(images):
    image_path = os.path.join(folder_path, image_file)
    actual_str = extract_actual_str(image_path)

    # perform image processing
    img = perform_image_processing(image_path)
    # update the image on new path and save it
    image_path = image_path[:-4] + "_processed.jpg"
    cv2.imwrite(image_path, img)

    # evaluate the fine-tuned models
    for model_name, model in zip(ocr_fine_tuned_models_names, models):
        my_results = model.ocr(image_path, cls=True)
        if my_results[0] is not None:
            predicted_str = ''.join(word_info[1] for word_info in my_results[0])
            accuracy = evaluate_image_accuracy(predicted_str, actual_str)
            print("predicted:", predicted_str)
            print("actual:", actual_str)
            print("accuracy:", accuracy)
            accuracy_scores[model_name].append(accuracy)
            for word_info in my_results[0]:
                # print("Detected Text:\n", word_info[1])
                print_char_confidence(word_info, my_results)
        else:
            accuracy_scores[model_name].append(0.0)  # No text detected case

    # evaluate the original PaddleOCR model
    paddle_results = paddle_ocr.ocr(image_path, cls=True)
    if paddle_results[0] is not None:
        predicted_str = ''.join(word_info[1] for word_info in paddle_results[0])
        accuracy = evaluate_image_accuracy(predicted_str, actual_str)
        accuracy_scores["PaddleOCR"].append(accuracy)
        # Display images with OpenCV
        # annotated_my_image = draw_results_on_image(image_path, paddle_results, "paddle_ocr")
        # annotated_my_image = cv2.cvtColor(annotated_my_image, cv2.COLOR_RGB2BGR)
        # cv2.imshow(f"Image segmentation", annotated_my_image)
        # print("Close the window or press 'q' to continue...")
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
    else:
        accuracy_scores["PaddleOCR"].append(0.0)


# Output the results
avg_scores, best_model, best_accuracy = calculate_avg_model_accuracy(accuracy_scores)
plot_accuracies_per_model(avg_scores)
print(f"The best model is {best_model} with an average accuracy of {best_accuracy}")


# Cleanup OpenCV windows
cv2.destroyAllWindows()

