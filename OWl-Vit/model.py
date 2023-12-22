from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class ObjectDetector:
    def __init__(self, model_path="google/owlvit-base-patch32", threshold=0.1):
        self.processor = OwlViTProcessor.from_pretrained(model_path)
        self.model = OwlViTForObjectDetection.from_pretrained(model_path)
        self.threshold = threshold
        self.image_tensor = None
        self.outputs = None
        self.boxes = None
        self.scores = None
        self.labels = None
        self.text = None

    def load_image(self, image_path):
        try:
            with Image.open(image_path) as image:
                self.image_tensor = torch.tensor(image).to(torch.float32) / 255.0
                self.image_tensor = self.image_tensor.unsqueeze(0)
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")

    def inference(self, texts):
        try:
            with torch.no_grad():
                inputs = self.processor(text=texts, images=self.image_tensor, return_tensors="pt")
                self.outputs = self.model(**inputs)
        except Exception as e:
            raise RuntimeError(f"Error during inference: {e}")

    def post_process(self, texts):
        target_sizes = torch.Tensor([self.image_tensor.size()[2:]])
        try:
            results = self.processor.post_process_object_detection(outputs=self.outputs, target_sizes=target_sizes, threshold=self.threshold)
            i = 0
            self.text = texts[i]
            self.boxes, self.scores, self.labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
            del results  # Remove unused variables to release memory
        except Exception as e:
            raise RuntimeError(f"Error during post-processing: {e}")

    def plot_boxes(self):
        fig, ax = plt.subplots(1)
        ax.imshow(self.image_tensor.squeeze(0).permute(1, 2, 0).numpy())
        for box, score, label in zip(self.boxes, self.scores, self.labels):
            box = [round(i, 2) for i in box.tolist()]
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
            ax.text(box[0], box[1], f"{self.text[label]}: {round(score.item(), 3)}", bbox=dict(facecolor="white", alpha=0.5))
        plt.axis("off")
        plt.show()
        plt.close()

if __name__ == "__main__":
    image_path = '/content/s-l1200.jpg'
    texts = [["a photo of all txt"]]

    object_detector = ObjectDetector()
    object_detector.load_image(image_path)
    object_detector.inference(texts)
    object_detector.post_process(texts)
    object_detector.plot_boxes()
