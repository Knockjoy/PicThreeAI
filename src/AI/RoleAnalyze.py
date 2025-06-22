# copyright (c) 2025 Yuuki Furuta

from PIL import Image
import json
import sys

sys.path.append("/root/picthree/PicThreeAI/src/AI")
import learn_clip

roles_dict = None
with open("/root/picthree/PicThreeAI/src/AI/AIconfig/Role/ChatGPT/roles_data30.json") as f:
    roles_dict = json.loads(f.read())

roles_data = []
all_data = []
data_lenth = []
roles = []

for j, i in roles_dict.items():
    temp = []
    temp.extend(i["image"].split(", "))
    temp.extend(i["shape_keywords"])
    temp.extend(i["associated_words"])
    roles_data.append([j, temp])
    all_data.extend([j])
    all_data.extend(temp)
    temp.extend([j])
    roles.append(j)
    data_lenth.append(len(temp))


# data_lenth
def split_by_lengths(data, lengths):
    result = []
    index = 0
    for length in lengths:
        result.append(sum(data[index : index + length]))
        index += length
    return result


def analyze(ImgPath:str)->str:
    img=Image.open(ImgPath)
    ImgAnalyzedData=learn_clip.analyzeImage(all_data,img)
    # ImgAnalyzedData["elements"][ImgAnalyzedData["score"].argmax()]
    roles_score=split_by_lengths(ImgAnalyzedData["score"],data_lenth)
    return roles[roles_score.index(max(roles_score))]

if __name__=="__main__":
    print(analyze("/root/picthree/PicThreeAI/Apple.png"))
    print(analyze("/root/picthree/PicThreeAI/canvas (4).png"))