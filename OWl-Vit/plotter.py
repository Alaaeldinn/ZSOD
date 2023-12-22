import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_boxes(image_tensor, boxes, scores, labels, text):
    fig, ax = plt.subplots(1)
    ax.imshow(image_tensor.squeeze(0).permute(1, 2, 0).numpy())
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        ax.text(box[0], box[1], f"{text[label]}: {round(score.item(), 3)}", bbox=dict(facecolor="white", alpha=0.5))
    plt.axis("off")
    plt.show()
    plt.close()
