import cv2
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BatchLoader:
    def __init__(self, image_path, batch_size, stride):
        self.image_path = image_path
        self.batch_size = batch_size
        self.stride = stride
        self.img_tensor = self._load_image()

    def _load_image(self):
        img = cv2.imread(self.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

    def image_to_batches(self):
        channels, height, width = self.img_tensor.shape

        i_indices = np.arange(0, height - self.batch_size + 1, self.stride)
        j_indices = np.arange(0, width - self.batch_size + 1, self.stride)

        i_strided, j_strided = np.meshgrid(i_indices, j_indices, indexing='ij')
        batch_indices = np.stack((i_strided, j_strided), axis=-1)

        batches = [self.img_tensor[:, i:i+self.batch_size, j:j+self.batch_size] for i, j in batch_indices.reshape(-1, 2)]

        return batches

    def plot_batches_on_image(self, batches):
        num_batches = len(batches)
        channels, height, width = self.img_tensor.shape

        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        ax.imshow(np.transpose(self.img_tensor.numpy(), (1, 2, 0)).astype(np.uint8))

        batch_size = batches[0].shape[1]

        for batch in batches:
            for j in range(0, height - batch_size + 1, self.stride):
                for k in range(0, width - batch_size + 1, self.stride):
                    rect = patches.Rectangle((k, j), batch_size, batch_size, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

        plt.axis('off')
        plt.show()

# Example usage
image_path = "path/to/your/image.jpg"
batch_size = 64
stride = 32

batch_loader = BatchLoader(image_path, batch_size, stride)
result_batches = batch_loader.image_to_batches()

print(f"Number of batches: {len(result_batches)}")
print(f"Shape of the first batch: {result_batches[0].shape}")

batch_loader.plot_batches_on_image(result_batches)
