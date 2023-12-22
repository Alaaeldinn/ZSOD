from PIL import Image
from plotter import plot_boxes
from owlvit_model import OWLVIT

if __name__ == "__main__":
    # Load the image directly as a PIL Image
    image_path = '/content/s-l1200.jpg'
    image = Image.open(image_path)

    prompts = [["a photo of all txt"]]

    model = OWLVIT()
    boxes, scores, labels, text = model(image, prompts)

    # Use the plot function outside the class
    plot_boxes(image, boxes, scores, labels, text)
