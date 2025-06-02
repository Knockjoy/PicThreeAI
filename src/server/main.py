from fastapi import FastAPI ,Request,File,UploadFile
from typing import List
import io 
from PIL import Image
import uvicorn
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from AI.learn_clip import analyzeImage,called

# from ..AI.learn_clip import analyzeImage

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/AIsample")
async def samp():
    try:
        r=called()
    except Exception as e:
        return {"errorValue":e} 
    else:
        return {"status":r}

@app.post("/Contents/analyzeImage")
async def Upload(
    element:List[str],
    imagefile:List[UploadFile]=File(None)
):
    try:
        bin_data=io.BytesIO(imagefile[0].file.read())
        img=Image.open(bin_data)
        result=analyzeImage(element,img)
    except Exception as e:
        return {"status":"failed","errorValue":e}
    return {"status":"success","data":result}

if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)