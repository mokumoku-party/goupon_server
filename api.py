from fastapi import FastAPI, UploadFile
import shutil
import goutouch_mediapipe as gm
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    )

@app.post("/check_gou_touch")
async def check_gou_touch(upload_file: UploadFile):
    path = f'upload_files/{upload_file.filename}'# api/filesディレクトリを作成しておく
    with open(path, 'wb+') as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    detection_result = gm.image_detector('upload_files/' + upload_file.filename)
    return gm.check_gou_touch(detection_result)

@app.get("/test")
async def test():
    return {"message":"hello api!"}
    