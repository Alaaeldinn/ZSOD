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

    def load_image(self, image_path):
        try:
            image = Image.open(image_path)
            image_array = torch.tensor(image).to(torch.float32) / 255.0
            image_array = image_array.unsqueeze(0)
            image.close()
            torch.cuda.empty_cache()
            return image_array
        except Exception as e:
            print(f"Error loading image: {e}")
            exit(1)

    def inference(self, texts, image_array):
        try:
            with torch.no_grad():
                inputs = self.processor(text=texts, images=image_array, return_tensors="pt")
                outputs = self.model(**inputs)
            torch.cuda.empty_cache()
            return outputs
        except Exception as e:
            print(f"Error during inference: {e}")
            exit(1)

    def post_process(self, outputs, image_array):
        target_sizes = torch.Tensor([image_array.size()[2:]])
        try:
            results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=self.threshold)
            i = 0
            text = texts[i]
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
            del image_array, inputs, outputs, results
            torch.cuda.empty_cache()
            return boxes, scores, labels, text
        except Exception as e:
            print(f"Error during post-processing: {e}")
            exit(1)

    def plot_boxes(self, image, boxes, scores, labels, text):
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
            ax.text(box[0], box[1], f"{text[label]}: {round(score.item(), 3)}", bbox=dict(facecolor="white", alpha=0.5))
        plt.axis("off")
        plt.show()
        plt.close()

if __name__ == "__main__":
    image_path = '/content/s-l1200.jpg'
    texts = [["a photo of all txt"]]

    object_detector = ObjectDetector()

    image_array = object_detector.load_image(image_path)
    outputs = object_detector.inference(texts, image_array)
    boxes, scores, labels, text = object_detector.post_process(outputs, image_array)
    object_detector.plot_boxes(image_array.squeeze(0).permute(1, 2, 0).numpy(), boxes, scores, labels, text)
