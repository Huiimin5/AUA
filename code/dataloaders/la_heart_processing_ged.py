import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nrrd
import os
output_size =[112, 112, 80]

def covert_h5(from_dir, to_dir, with_norm=True):
    # listt = glob('../../LA_dataset/2018LA_Seg_TrainingSet/*/lgemri.nrrd')
    listt = glob(os.path.join(from_dir, '*/lgemri.nrrd'))
    for item in tqdm(listt):
        image, img_header = nrrd.read(item)
        label, gt_header = nrrd.read(item.replace('lgemri.nrrd', 'laendo.nrrd'))
        label = (label == 255).astype(np.uint8)
        mask = (image > 0).astype(np.uint8)
        w, h, d = label.shape

        tempL = np.nonzero(label)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        if with_norm:
            image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        image = image[minx:maxx, miny:maxy]
        label = label[minx:maxx, miny:maxy]
        mask = mask[minx:maxx, miny:maxy]
        print(label.shape)
        new_path = item.replace(from_dir, to_dir).replace('lgemri.nrrd', 'mri_norm2.h5')
        print(new_path)
        new_dir = os.path.dirname(new_path)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        f = h5py.File(new_path, 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.create_dataset('mask', data=mask, compression="gzip")
        f.close()

if __name__ == '__main__':

    from_dir_train = '~/dataset/2018LA_Seg_TrainingSet'
    to_dir_train = '../../data_3items_no_norm/2018LA_Seg_TrainingSet'
    covert_h5(from_dir_train, to_dir_train, False)
