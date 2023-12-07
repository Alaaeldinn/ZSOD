import torch
import matplotlib.pyplot as plt
import numpy as np
# Define colors for bounding boxes
colors = ['#FAFF00', '#8CF1FF']

def generate_patches(image, patch_size=256):
    """
    Generates patches from the input image.
    """
    # Add an extra dimension for later calculations
    img_patches = image.data.unfold(0, 3, 3)
    # Break the image into patches (in the height dimension)
    img_patches = img_patches.unfold(1, patch_size, patch_size)
    # Break the image into patches (in the width dimension)
    img_patches = img_patches.unfold(2, patch_size, patch_size)
    return img_patches

def calculate_scores(img_patches, prompt, window=6, stride=1):
    """
    Calculates similarity scores for image patches based on a given prompt.
    """
    # Initialize scores and runs arrays
    scores = torch.zeros(img_patches.shape[1], img_patches.shape[2])
    runs = torch.ones(img_patches.shape[1], img_patches.shape[2])

    # Iterate through patches
    for Y in range(0, img_patches.shape[1] - window + 1, stride):
        for X in range(0, img_patches.shape[2] - window + 1, stride):
            # Initialize an array to store big patches
            big_patch = torch.zeros(patch * window, patch * window, 3)
            # Get a single big patch
            patch_batch = img_patches[0, Y:Y + window, X:X + window]
            # Iteratively build all big patches
            for y in range(window):
                for x in range(window):
                    big_patch[y * patch:(y + 1) * patch, x * patch:(x + 1) * patch, :] = patch_batch[y, x].permute(1, 2, 0)
            inputs = processor(
                images=big_patch,  # Image transmitted to the model
                return_tensors="pt",  # Return PyTorch tensor
                text=prompt,  # Text transmitted to the model
                padding=True
            ).to(device)  # Move to device if possible

            score = model(**inputs).logits_per_image.item()
            # Sum up similarity scores
            scores[Y:Y + window, X:X + window] += score
            # Calculate the number of runs
            runs[Y:Y + window, X:X + window] += 1
    # Calculate average scores
    scores /= runs
    # Clip scores
    for _ in range(3):
        scores = np.clip(scores - scores.mean(), 0, np.inf)
    # Normalize scores
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores

def calculate_box(scores, patch_size=256, threshold=0.5):
    """
    Calculates the bounding box based on the detected scores.
    """
    detection = scores > threshold
    # Find box corners
    y_min, y_max = np.nonzero(detection)[:, 0].min().item(), np.nonzero(detection)[:, 0].max().item() + 1
    x_min, x_max = np.nonzero(detection)[:, 1].min().item(), np.nonzero(detection)[:, 1].max().item() + 1
    # Convert from patch coordinates to pixel coordinates
    y_min *= patch_size
    y_max *= patch_size
    x_min *= patch_size
    x_max *= patch_size
    # Calculate box height and width
    height = y_max - y_min
    width = x_max - x_min
    return x_min, y_min, width, height

def object_detection(prompts, image, patch_size=256, window=6, stride=1, threshold=0.5):
    """
    Performs object detection on the input image using the given prompts.
    """
    # Build image patches for detection
    img_patches = generate_patches(image, patch_size)
    # Convert image to a format for displaying with matplotlib
    image_array = np.moveaxis(image.data.numpy(), 0, -1)
    # Initialize a plot to display the image + bounding boxes
    fig, ax = plt.subplots(figsize=(Y * 0.5, X * 0.5))
    ax.imshow(image_array)
    # Process image through object detection steps
    for i, prompt in enumerate(tqdm(prompts)):
        scores = calculate_scores(img_patches, prompt, window, stride)
        x, y, width, height = calculate_box(scores, patch_size, threshold)
        # Create the bounding box
        rect = patches.Rectangle((x, y), width, height, linewidth=3, edgecolor=colors[i], facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()
