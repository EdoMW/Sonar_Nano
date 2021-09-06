import os
import shutil

source_path = r'C:\Users\Administrator\Downloads\masks'
destination_path = r'C:\Users\Administrator\Downloads\masks_npy'
source = os.listdir(source_path)
for file in source:
    if file.endswith('.npy'):
        print(file)
        shutil.move(os.path.join(source_path, file), destination_path)

import numpy as np
# for i in range(len(source)):
#     file_name = source[i]
#     file_name = os.path.join(source_path, file_name)
#     a = np.load(file_name)
#     mask_npy = a.f.arr_0
#     print(mask_npy.shape)
#     dir_path = os.path.join(destination_path, (file_name[:-4] + '.npy'))
#     np.save(dir_path, mask_npy)