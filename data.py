import pandas as pd
import numpy as np
import config
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class FacialKeyPointDataset(Dataset):
    def __init__(self, csv_file, train=True, transform=None):
        super().__init__()
        df = pd.read_csv(csv_file)
        #df.drop(columns="Unnamed: 0", inplace=True)
        self.data = df
        self.category_names = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'left_eye_inner_corner_x', 'left_eye_inner_corner_y', 'left_eye_outer_corner_x', 'left_eye_outer_corner_y', 'right_eye_inner_corner_x', 'right_eye_inner_corner_y', 'right_eye_outer_corner_x', 'right_eye_outer_corner_y', 'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y', 'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y', 'mouth_center_top_lip_x', 'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        if self.train:
            image = np.array(self.data.iloc[index, 31].split()).astype(np.float32)
            labels = np.array(self.data.iloc[index, 1:31].tolist())
            labels[np.isnan(labels)] = -1
        else:
            image = np.array(self.data.iloc[index, 1].split()).astype(np.float32)
            labels = np.zeros(30)

        ignore_indices = labels == -1
        labels = labels.reshape(15,2)

        if self.transform:
            image = np.repeat(image.reshape(96,96,1), 3, axis=2).astype(np.uint8)
            augmentations = self.transform(image=image, keypoints=labels)
            image = augmentations['image']
            labels = augmentations['keypoints']

        labels = np.array(labels).reshape(-1)
        labels[ignore_indices] = -1

        return image, labels
    
if __name__ == "__main__":
    dataset = FacialKeyPointDataset(csv_file="data/train_15.csv", train=True, transform=config.val_transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for idx, (x, y) in enumerate(loader):
        plt.imshow(x[0][0].detach().cpu().numpy(), cmap='gray')
        plt.plot(y[0][0::2].detach().cpu().numpy(), y[0][1::2].detach().cpu().numpy(), "go")
        plt.show()