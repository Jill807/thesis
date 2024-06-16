import pandas as pd
from PIL import Image
import PIL


data = pd.read_parquet('src/data/time_dataset_no_time_encoding.parquet', engine = 'fastparquet')
count = 0
idx_list = []


for idx in range(len(data)):
    try:
        image = Image.open(data.loc[idx, 'FilePath'])

    except (OSError, PIL.UnidentifiedImageError):
        print("corrupted:", idx, data.loc[idx, 'rounded_datetime'])
        if count%10 == 0:
            print(count)
        count+=1
        idx_list.append(idx)
print(min(idx_list), max(idx_list))
print(count)

data = data.drop(index = idx_list)
data = data.drop(columns = ["level_0", "index"])
data = data.reset_index(drop=True)
print(data.tail(20))
data.reset_index()
print(len(data))
print(data.tail(20))
data.to_parquet('src/data/dataset_without_corrupted_images.parquet', engine= 'fastparquet')