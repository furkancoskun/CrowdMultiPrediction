import os
import cv2
import numpy as np
import yaml
import random
import json
from scipy.ndimage.filters import gaussian_filter
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from Utils import im_to_torch

sample_random = random.Random()

class CMP_Dataset(Dataset):
    def __init__(self, cfg, train=True):
        super(CMP_Dataset, self).__init__()

        self.color = cfg["AUGMENTATION"]["COLOR"]
        self.flip = cfg["AUGMENTATION"]["FLIP"]
        self.rotation = cfg["AUGMENTATION"]["ROTATION"]
        self.gray = cfg["AUGMENTATION"]["GRAY"]
        self.blur = cfg["AUGMENTATION"]["BLUR"]

        """self.transform_extra = transforms.Compose(
            [transforms.ToPILImage(), ] +
            ([transforms.ColorJitter(0.05, 0.05, 0.05, 0.05), ] if self.color > random.random() else [])
            + ([transforms.RandomHorizontalFlip(), ] if self.flip > random.random() else [])
            + ([transforms.RandomRotation(degrees=10), ] if self.rotation > random.random() else [])
            + ([transforms.Grayscale(num_output_channels=3), ] if self.gray > random.random() else [])
        )"""

        dataset_path = cfg["DATASETS"]["GTA_EVENTS_DATASET"]["PATH"]
        if (train):
            txt_path = cfg["DATASETS"]["GTA_EVENTS_DATASET"]["TRAIN_TXT"]
        else:
            txt_path = cfg["DATASETS"]["GTA_EVENTS_DATASET"]["TEST_TXT"]
        
        self.lstm_seq_frame_count = cfg["LSTM_SEQUENCE_FRAME_COUNT"]

        self.sequences = []
        txt_file = open(txt_path)
        _ = txt_file.readline()
        lines = txt_file.readlines()
        for line in lines:
            video_name, anomaly_frame, anomaly_frame_amount = line.split(',')
            anomaly_frame = int(anomaly_frame)
            anomaly_frame_amount = int(anomaly_frame_amount)
            seq_directory = os.path.join(dataset_path,video_name,video_name)

            frames = []
            for i in range (anomaly_frame-anomaly_frame_amount, anomaly_frame):
                image_name = str(i).zfill(10)
                frame_path = os.path.join(seq_directory, image_name + ".tiff")
                json_file_path = os.path.join(seq_directory, image_name + ".json")
                count=0
                coord_list=[]
                with open(json_file_path) as f:
                    json_file = json.load(f)
                    for j in json_file['Detections']:
                        if "IK_Head" not in j["Bones"]:
                            continue
                        count = count +1
                        x_coord = round(j["Bones"]["IK_Head"]["X"]*2560)
                        y_coord = round(j["Bones"]["IK_Head"]["Y"]*1440)
                        coord_list.append({'x':x_coord, 'y':y_coord})
                dict = {
                    "video_name": video_name,
                    "frame_path": frame_path,
                    "anomaly": False,
                    "person_coord_list": coord_list,
                    "person_count": count
                }
                frames.append(dict)

            for i in range (anomaly_frame, anomaly_frame+anomaly_frame_amount):
                image_name = str(i).zfill(10)
                frame_path = os.path.join(seq_directory, image_name + ".tiff")
                json_file_path = os.path.join(seq_directory, image_name + ".json")
                count=0
                coord_list=[]
                with open(json_file_path) as f:
                    json_file = json.load(f)
                    for j in json_file['Detections']:
                        if "IK_Head" not in j["Bones"]:
                            continue
                        count = count +1
                        x_coord = round(j["Bones"]["IK_Head"]["X"]*2560)
                        y_coord = round(j["Bones"]["IK_Head"]["Y"]*1440)
                        coord_list.append({'x':x_coord, 'y':y_coord})
                dict = {
                    "video_name": video_name,
                    "frame_path": frame_path,
                    "anomaly": True,
                    "person_coord_list": coord_list,
                    "person_count": count
                }
                frames.append(dict)

            for i in range(self.lstm_seq_frame_count, len(frames)):
                self.sequences.append(frames[i-self.lstm_seq_frame_count:i])
            
        sample_random.shuffle(self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        seq_count = len(seq)
        frames =[]
        count_gts = []
        anomaly_count=0
        for i in range(seq_count):
            frame = cv2.imread(seq[i]["frame_path"])
            frame = im_to_torch(frame)
            frames.append(frame)
            count_gts.append(seq[i]["person_count"])
            if seq[i]["anomaly"] : anomaly_count = anomaly_count+1 

        if (anomaly_count > (seq_count/2)):
            anomaly_gt = True
        else:
            anomaly_gt = False

        return frames, count_gts, anomaly_gt

if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader

    yaml_name = "train.yaml"
    yaml_file = open(yaml_name, 'r')
    cfg = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)

    train_set = CMP_Dataset(cfg, train=False)

    train_loader = DataLoader(train_set, batch_size=1, num_workers=0, pin_memory=False)

    print("Data loader created!")

    cv2.namedWindow("images", cv2.WINDOW_NORMAL)
    for iter, data in enumerate(train_loader):
        print(f"Iter: {iter}")
        frames, count_gts, anomaly_gt = data
        images = []
        for frame in frames:
            frame = frame.cpu().numpy()[0,:,:,:].transpose(1,2,0).astype(np.uint8) 
            if (images == []):
                images = frame
            else:
                images = cv2.hconcat([images, frame])
        for count in count_gts:
            print ("Count: " + str(count.cpu()))
        print ("Anomaly: " + str(anomaly_gt.cpu()))
        cv2.imshow("images", images) 
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break
    