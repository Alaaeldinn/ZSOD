from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from plotter import plot_boxes

class OWLVIT:
    def __init__(self, model_path="google/owlvit-base-patch32", threshold=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OwlViTProcessor.from_pretrained(model_path)
        self.model = OwlViTForObjectDetection.from_pretrained(model_path).to(self.device)
        self.threshold = threshold
        self.image_tensor = None
        self.outputs = None
        self.text = None

    def __call__(self, image, texts):
        self.load_image(image)
        self.inference(texts)
        return self.post_process(texts)

    def load_image(self, image):
        with torch.no_grad():
            try:
                image_tensor = torch.tensor(image, dtype=torch.float32).to(self.device) / 255.0
                self.image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            except Exception as e:
                raise ValueError(f"Error loading image: {e}")

    def inference(self, texts):
        with torch.no_grad():
            try:
                inputs = self.processor(text=texts, images=self.image_tensor, return_tensors="pt")
                self.outputs = self.model(**inputs)
            except Exception as e:
                raise RuntimeError(f"Error during inference: {e}")

    def post_process(self, texts):
        target_size = torch.tensor([self.image_tensor.size()[2:]], device=self.device)
        try:
            result = self.processor.post_process_object_detection(outputs=self.outputs, target_sizes=target_size, threshold=self.threshold)
            self.text = texts[0]
            return result[0]["boxes"], result[0]["scores"], result[0]["labels"], self.text
        except Exception as e:
            raise RuntimeError(f"Error during post-processing: {e}")

if __name__ == "__main__":
    # Load the image directly as a PIL Image
    image_path = '/content/s-l1200.jpg'
    image = Image.open(image_path)

    prompts = [["a photo of all txt"]]

    model = OWLVIT()
    boxes, scores, labels, text = model(image, prompts)

    # Use the plot function outside the class
    plot_boxes(image, boxes, scores, labels, text)
