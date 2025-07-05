from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import io
from predic import predict_image,Y11_N,Y11_S

app = FastAPI()


@app.post("/predict_nano")
async def predict_nano(file: UploadFile = File(...)):
    img = Image.open(file.file)

    predicted_img=predict_image(Y11_N,img)

    buf = io.BytesIO()
    predicted_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")


@app.post("/predict_small")
async def predict_small(file: UploadFile = File(...)):
    img = Image.open(file.file)

    predicted_img = predict_image(Y11_S,img)

    buf = io.BytesIO()
    predicted_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")


