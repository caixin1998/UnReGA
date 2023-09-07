import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch
import pathlib
import random
import torchvision.transforms as transforms

from PIL import Image
def gazeto2d(gaze):
    yaw = np.arctan2(-gaze[0], -gaze[2])
    pitch = np.arcsin(-gaze[1])
    return np.array([yaw, pitch])



def get_transform(grayscale=False, convert=True, crop = False):
    transform_list = []
    transform_list += [transforms.ToPILImage()]
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if crop:
        transform_list += [transforms.CenterCrop(192)]
        transform_list += [transforms.Resize(224)]
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    return transforms.Compose(transform_list)

class loader(Dataset):
    def __init__(self, path, root, pic_num, header=True, target = "mpii"):
        self.lines = []
        self.pic_num = pic_num
        self.target = target
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    line = f.readlines()
                    if header: line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                self.lines = f.readlines()
                if header: self.lines.pop(0)
        if self.pic_num >= 0:
            self.lines = self.lines[:self.pic_num]
        self.root = pathlib.Path(root)
        self.transform = get_transform()

    def __len__(self):
        # if self.pic_num < 0:
        return len(self.lines)
        # return self.pic_num

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")
        # print(line)

        # name = line[0].split('/')[0]
        name = line[0]
        # if self.target == "mpii":
        #     gaze2d = line[7]
        #     head2d = line[8]
        # else:  
        gaze2d = line[1]
        head2d = line[2] 
     
        # lefteye = line[1]
        # righteye = line[2]
        face = line[0]
        # if self.target == "mpii":
        #     label = np.array(gaze2d.split(",")[::-1]).astype("float")
        # else:
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label[:2]).type(torch.FloatTensor)
        # print(label.shape)
        headpose = np.array(head2d.split(",")).astype("float")
        headpose = torch.from_numpy(headpose[:2]).type(torch.FloatTensor)

        # rimg = cv2.imread(os.path.join(self.root, righteye))/255.0
        # rimg = rimg.transpose(2, 0, 1)

        # limg = cv2.imread(os.path.join(self.root, lefteye))/255.0
        # limg = limg.transpose(2, 0, 1)

        # print(self.root/name/ face)
        imgpath = str(self.root / face)
        if self.target[:3] == "mix": imgpath = face
        fimg = cv2.imread(imgpath)

        ycrcb = cv2.cvtColor(fimg, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        fimg = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        # fimg=crop(fimg)
        # print(fimg.shape)
        # fimg = cv2.resize(fimg, (448, 448)) / 255.0

        fimg = self.transform(fimg)
        img = {"face": fimg,
               "head_pose": headpose,
               "name": name}

        # img = {"left":torch.from_numpy(limg).type(torch.FloatTensor),
        #        "right":torch.from_numpy(rimg).type(torch.FloatTensor),
        #        "face":torch.from_numpy(fimg).type(torch.FloatTensor),
        #        "head_pose":headpose,
        #        "name":name}
        return img, label


def txtload(labelpath, imagepath, batch_size, pic_num=-1, shuffle=True, num_workers=0, header=True, target = 1):
    # print(labelpath,imagepath)
    dataset = loader(labelpath, imagepath, pic_num, header, target = target)
    print(f"[Read Data]: Total num: {len(dataset)}")
    # print(f"[Read Data]: Label path: {labelpath}")
    # print(dataset.lines[:10])
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # seed_everything(1)
    path = '/home/caixin/GazeData/MPIIFaceGaze/Label/p00.label'
    d = txtload(path, '/home/caixin/GazeData/MPIIFaceGaze/Image', batch_size=32, pic_num=5,
                shuffle=False, num_workers=4, header=True)
    print(len(d))
    for i, (img, label) in enumerate(d):
        print(i, label)
