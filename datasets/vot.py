from torch.utils.data import *
import Image
import os

class VotDataset(Dataset):
    def __init__(self, root, seqs, train=True, transform=None, target_transform=None, num_frames=5):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.num_frames = num_frames
        self.total_len = 0

        if (self.train):
            self.path = os.path.join(root, '/train')
        else:
            self.path = os.path.join(root, '/test')

        self.seqs = []
        for seq in seqs:
            seq_path = os.path.join(self.path, seq)
            images = []
            gts = []

            for file in os.listdir(seq_path):
                if os.path.splitext(file) == 'jpg':
                    images.append(file)
                elif os.path.split(file) == 'gt.txt':
                    with open(file) as f:
                        truths = []

                        for line in f.readlines():
                            arr = line.strip('\n').split(',')
                            coords = []

                            for data in arr:
                                if data == 'NaN':
                                    coords.append(-1)
                                else:
                                    coords.append(int(data))

                            if coords[0] == -1:
                                truths.append([])
                            else:
                                truths.append(coords)

                        gts.append(truths)
            
            last_len = self.total_len
            self.total_len += len(images) - (self.num_frames - 1)
            self.seqs.append((images.sort(), gts, last_len, self.total_len))

    def __getitem__(self, index):
        for images, gts, last_count, count in self.seqs:
            if (last_count <= index) and (index < count):
                local_index = index - last_count

                # read num_frames of pics from local_index
                image_files = images[local_index:local_index + self.num_frames]
                image = []

                for image_file in image_files:
                    im = Image.open(image_file)
                    image.append(im)

                if self.transform is not None:
                    im = self.transform(im)

                # read num_frames of data from gt
                gt = gts[local_index:local_index + self.num_frames]

                if self.target_transform is not None:
                    gt = self.target_transform(gt)

                return image, gt

    def __len__(self):
        return self.total_len
