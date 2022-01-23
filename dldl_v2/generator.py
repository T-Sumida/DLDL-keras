import cv2
import albumentations as A
import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence
from typing import Tuple, List

MAX_AGE = 100

def calc_normal_distribution(x, mu, sigma=1):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-np.square(x - mu) / (2 * np.square(sigma)))

class ImageGenerator(Sequence):
    def __init__(self, batch_size: int, csv_path: str, transfomer: A.Compose) -> None:
        """Initialize

        Args:
            batch_size (int): バッチサイズ
            csv_path (str): CSVファイルパス
            transfomer (A.Compose): 変換処理関数
        """
        self.df = pd.read_csv(csv_path)
        self.batch_size = batch_size
        self.indices = np.arange(len(self.df))
        self.transformer = transfomer

        np.random.shuffle(self.indices)

    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], Tuple[List[np.ndarray], List[float]]]:
        imgs, ages, age_dists = [], [], []
        sample_indices = self.indices[index * self.batch_size:(index+1)*self.batch_size]

        for _, row in self.df.iloc[sample_indices].iterrows():
            age_dist = [0] * MAX_AGE
            file_path = str(row["file"])
            age = int(row["age"])

            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = self.transformer(image=img)["image"]
            age_dist = np.array([calc_normal_distribution(i, age) for i in range(0, MAX_AGE)])

            imgs.append(img)
            ages.append(age / MAX_AGE)
            age_dists.append(age_dist)
        
        return imgs, (age_dists, ages)

    def __len__(self) -> int:
        return len(self.df) // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
