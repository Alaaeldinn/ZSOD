import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.patches as patches


class zsed:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.colors = ['#FAFF00', '#8CF1FF']

    def generate_patches(self, image, patch_size=256):
        img_patches = image.data.unfold(0, 3, 3)
        img_patches = img_patches.unfold(1, patch_size, patch_size)
        img_patches = img_patches.unfold(2, patch_size, patch_size)
        return img_patches

    def calculate_scores(self, img_patches, prompt, window=6, stride=1):
        scores = torch.zeros(img_patches.shape[1], img_patches.shape[2])
        runs = torch.ones(img_patches.shape[1], img_patches.shape[2])

        for Y in range(0, img_patches.shape[1] - window + 1, stride):
            for X in range(0, img_patches.shape[2] - window + 1, stride):
                big_patch = torch.zeros(patch_size * window, patch_size * window, 3)
                patch_batch = img_patches[0, Y:Y + window, X:X + window]

                for y in range(window):
                    for x in range(window):
                        big_patch[y * patch_size:(y + 1) * patch_size, x * patch_size:(x + 1) * patch_size, :] = patch_batch[y, x].permute(1, 2, 0)

                inputs = self.processor(
                    images=big_patch,
                    return_tensors="pt",
                    text=prompt,
                    padding=True
                ).to(self.device)

                score = self.model(**inputs).logits_per_image.item()
                scores[Y:Y + window, X:X + window] += score
                runs[Y:Y + window, X:X + window] += 1

        scores /= runs

        for _ in range(3):
            scores = np.clip(scores - scores.mean(), 0, np.inf)

        scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores

    def bounding_box(self, scores, patch_size=256, threshold=0.5):
        detection = scores > threshold
        y_min, y_max = np.nonzero(detection)[:, 0].min().item(), np.nonzero(detection)[:, 0].max().item() + 1
        x_min, x_max = np.nonzero(detection)[:, 1].min().item(), np.nonzero(detection)[:, 1].max().item() + 1

        y_min *= patch_size
        y_max *= patch_size
        x_min *= patch_size
        x_max *= patch_size

        height = y_max - y_min
        width = x_max - x_min
        return x_min, y_min, width, height

    def detect(self, prompts, image, patch_size=256, window=6, stride=1, threshold=0.5):
        img_patches = self.generate_patches(image)
        image_array = np.moveaxis(image.data.numpy(), 0, -1)

        fig, ax = plt.subplots(figsize=(img_patches.shape[2] * 0.5, img_patches.shape[1] * 0.5))
        ax.imshow(image_array)

        for i, prompt in enumerate(tqdm(prompts)):
            scores = self.calculate_scores(img_patches, prompt, window, stride)
            x, y, width, height = self.bounding_box(scores, patch_size, threshold)

            rect = patches.Rectangle((x, y), width, height, linewidth=3, edgecolor=self.colors[i], facecolor='none')
            ax.add_patch(rect)

        plt.show()

