from PIL import Image
from lang_sam.lang_sam import LangSAM

if __name__ == '__main__':
    model = LangSAM()
    image_path = 'test.jpg'
    image = Image.open(image_path).convert("RGB")
    boxes, logits, phrases = model.predict_dino(image, "building", box_threshold=0.3, text_threshold=0.25)
    print(boxes)
