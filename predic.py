from ultralytics import YOLO
from PIL import Image


Y11_N = YOLO(r"Notebooks\runs\detect\train\weights\best.pt")

Y11_S = YOLO(r"Notebooks\runs\detect\train2\weights\best.pt")

def predict_image(model,img):
    result= model.predict(
    
    img
    
    )
    result = result[0]
    
    img = result.plot()
    
    pil_image = Image.fromarray(img)
    
    return pil_image

if __name__ == "__main__":

    predicted_image=predict_image(r'\Data\valid\images\14_jpg.rf.7183b1cf7b55ac0f801099e3936964a6.jpg')
    predicted_image.show()
