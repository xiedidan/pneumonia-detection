import numpy as np

# convert (xmin, ymin, xmax, ymax) to (xcenter, ycenter, w, h)
def bbox2center(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    return (
        w / 2 + bbox[0],
        h / 2 + bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1]
    )

class ClassCounter(object):
    def __init__(self, classes):
        self.classes = {}

        for cl in classes:
            self.classes[cl] = 0

    def inc(self, className):
        self.classes[className] += 1

    def get_result(self):
        return self.classes

class BoxCounter(object):
    def __init__(self):
        self.counter = {}

    def inc(self, bin):
        if bin not in self.counter.keys():
            self.counter[bin] = 1
        else:
            self.counter[bin] += 1

    def get_result(self):
        return self.counter

class DetectionCounter(object):
    def __init__(self):
        self.xs = []
        self.ys = []
        self.levels = []
        self.ratios = []

    def inc(self, bboxes):
        for bbox in bboxes:
            x, y, w, h = bbox2center(bbox.numpy())

            self.xs.append(x)
            self.ys.append(y)

            if w < h:
                self.ratios.append(-1. * h / w)
                self.levels.append(w * w)
            elif w > h:
                self.ratios.append(w / h)
                self.levels.append(h * h)
            elif (1. - w > 0.00001) or (1. - h > 0.0001):
                self.ratios.append(w / h)
                self.levels.append(w * h) 

    def get_result(self):
        return self.xs, self.ys, self.ratios, self.levels

class MeanAndStd(object):
    def __init__(self):
        self.mean = 0.
        self.std = 0.
        self.total = 0
    
    def inc(self, images):
        self.total += len(images)

        for image in images:
            np_image = image.numpy()
            self.mean += np.mean(np_image)
            self.std += np.std(np_image)

    def get_result(self):
        return self.mean / self.total, self.std / self.total
