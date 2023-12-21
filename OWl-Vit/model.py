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
        self.image_array = None
        self.outputs = None
        self.boxes = None
        self.scores = None
        self.labels = None
        self.text = None

    def load_image(self, image_path):
        try:
            image = Image.open(image_path)
            self.image_array = torch.tensor(image).to(torch.float32) / 255.0
            self.image_array = self.image_array.unsqueeze(0)
            image.close()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error loading image: {e}")
            exit(1)

    def inference(self, texts):
        try:
            with torch.no_grad():
                inputs = self.processor(text=texts, images=self.image_array, return_tensors="pt")
                self.outputs = self.model(**inputs)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during inference: {e}")
            exit(1)

    def post_process(self):
        target_sizes = torch.Tensor([self.image_array.size()[2:]])
        try:
            results = self.processor.post_process_object_detection(outputs=self.outputs, target_sizes=target_sizes, threshold=self.threshold)
            i = 0
            self.text = texts[i]
            self.boxes, self.scores, self.labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
            del results  # Remove unused variables to release memory
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during post-processing: {e}")
            exit(1)

    def plot_boxes(self):
        fig, ax = plt.subplots(1)
        ax.imshow(self.image_array.squeeze(0).permute(1, 2, 0).numpy())
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
    object_detector.post_process()
    object_detector.plot_boxes()
