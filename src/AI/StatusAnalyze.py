# copyright (c) 2025 Yuuki Furuta

from PIL import Image
import json
import sys

sys.path.append("/root/picthree/PicThreeAI/src/AI")
import learn_clip

roles_data = []
all_data = []
data_lenth = []
status = []

db="/root/picthree/PicThreeAI/src/AI/AIconfig/Status/ChatGPT/Status_data50.json"
status_dict = None
with open(db) as f:
    status_dict = json.loads(f.read())
    

for i,j in status_dict.items():
    temp=[]
    temp.extend(j)
    status.append(i)
    all_data.extend([i])
    all_data.extend(temp)
    data_lenth.append(len(temp))
    pass

def split_by_lengths(data, lengths):
    result = []
    index = 0
    for length in lengths:
        result.append(sum(data[index : index + length]))
        index += length
    return result

def analyze(imgPath:str):
    """
    return [hp,attack,defence,speed]
    """
    img=Image.open(imgPath)
    ImgAnalyzedData=learn_clip.analyzeImage(all_data,img)
    status_score=split_by_lengths(ImgAnalyzedData["score"],data_lenth)
    return list(map(float,status_score))

if __name__=="__main__":
    print(analyze("/root/picthree/PicThreeAI/Apple.png"))