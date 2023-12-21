from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Combine import statements
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

def plot_boxes(image, boxes, scores, labels, text):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        ax.text(box[0], box[1], f"{text[label]}: {round(score.item(), 3)}", bbox=dict(facecolor="white", alpha=0.5))
    plt.axis("off")
    plt.show()
    plt.close()  # Close the plot to release memory

# Load image with error handling and release memory
image_path = '/content/s-l1200.jpg'
try:
    image = Image.open(image_path)
    image_array = torch.tensor(image).to(torch.float32) / 255.0  # Convert image to torch tensor
    image_array = image_array.unsqueeze(0)  # Add batch dimension
    image.close()  # Close the image to release memory
    torch.cuda.empty_cache()  # Release GPU memory
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

# Process image and perform inference with error handling and memory release
texts = [["a photo of all txt"]]
try:
    with torch.no_grad():
        inputs = processor(text=texts, images=image_array, return_tensors="pt")
        outputs = model(**inputs)
    torch.cuda.empty_cache()  # Release GPU memory
except Exception as e:
    print(f"Error during inference: {e}")
    exit(1)

# Post-process object detection outputs with error handling and memory release
target_sizes = torch.Tensor([image_array.size()[2:]])
try:
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    del image_array, inputs, outputs, results  # Remove unused variables to release memory
    torch.cuda.empty_cache()  # Release GPU memory
except Exception as e:
    print(f"Error during post-processing: {e}")
    exit(1)

# Plot bounding boxes with error handling and memory release
try:
    plot_boxes(image, boxes, scores, labels, text)
    del boxes, scores, labels, text  # Remove unused variables to release memory
    torch.cuda.empty_cache()  # Release GPU memory
except Exception as e:
    print(f"Error during plotting: {e}")
    exit(1)