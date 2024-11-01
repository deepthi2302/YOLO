# YOLO
YOLO
# prompt: use yolo bulit model for object detection print the ouput plot it

!pip install ultralytics
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load a model
model = YOLO('yolov8n.pt')  # load an official model

# Run inference on an image
results = model('/content/WhatsApp Image 2024-11-01 at 07.37.04_2307035a.jpg') #replace with the image path

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(im)
    plt.show()

# Print the results
print(results[0].boxes) # print the bounding boxes
print(results[0].masks) # print the masks
print(results[0].keypoints) # print the keypoints
print(results[0].probs) # print the probabilities for each class

# prompt: use yolo bulit model for object detection print the ouput plot it

!pip install ultralytics
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load a model
model = YOLO('yolov8n.pt')  # load an official model

# Run inference on an image
results = model('/content/WhatsApp Image 2024-11-01 at 07.37.04_2307035a.jpg') #replace with the image path

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(im)
    plt.show()

# Print the results
print(results[0].boxes) # print the bounding boxes
print(results[0].masks) # print the masks
print(results[0].keypoints) # print the keypoints
print(results[0].probs) # print the probabilities for each class

<img width="456" alt="image" src="https://github.com/user-attachments/assets/5a654a14-1d04-42be-a534-116758de6753">
