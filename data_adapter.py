import caffe

from datasets.pneumonia import *
from datasets.transforms import *

# configs
num_classes = 2

# spawned workers on windows take too much gmem
number_workers = 8
if sys.platform == 'win32':
    number_workers = 2

size = 512
mean = [0.49043187350911405]
std = [0.22854086980778032]

classMapping = {
    'Normal': 0,
    'No Lung Opacity / Not Normal': 0,
    'Lung Opacity': 1
}

class PythonDataAdapter(caffe.Layer):
    def setup(self, bottom, top):
        assert len(top) == 2

        self.topNames=['data', 'label']

        # read parameters from `self.param_str`
        params = eval(self.param_str)

        if params['phase'] == 'train':
            trainTransform = Compose([
                RandomResizedCrop(size=size, p=0.7, scale=(0.9, 1.), ratio=(0.9, 1/0.9)),
                Percentage(size=size),
                ToTensor()
            ])

            trainSet = PneumoniaDetectionDataset(
                root=params['root'],
                phase='train',
                transform=trainTransform,
                classMapping=classMapping,
                num_classes=num_classes
            )
            self.batchLoader = torch.utils.data.DataLoader(
                trainSet,
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=number_workers,
                collate_fn=detectionCollate,
            )
        elif params['phase'] == 'val':
            valTransform = Compose([
                Resize(size=size),
                Percentage(size=size),
                ToTensor(),
            ])

            valSet = PneumoniaDetectionDataset(
                root=params['root'],
                phase='val',
                transform=valTransform,
                classMapping=classMapping,
                num_classes=num_classes
            )
            self.batchLoader = torch.utils.data.DataLoader(
                valSet,
                batch_size=params['batch_size'],
                shuffle=False,
                num_workers=number_workers,
                collate_fn=detectionCollate,
            )

        # reshape images
        top[0].reshape(params['batch_size'], 1, params['size'][0], params['size'][1])

        self.params = params

    def reshape(self, bottom, top):
        # only reshape gts online
        top[1].reshape(self.params['batch_size'], -1, 5)

    def forward(self, bottom, top):
        images, gts, ws, hs, ids = next(self.batchLoader)

        top[0].data[...] = images[...]
        top[1].data[...] = gts[...]

    def backward(self, top, propagate_down, bottom):
        pass
