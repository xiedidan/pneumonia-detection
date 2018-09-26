import torch

def export_classification_csv(file, ids, labels, classes, confs):
    for i in range(len(ids)):
        file.write('{},{},{},{}\n'.format(ids[i], labels[i], classes[i], confs[i]))

def export_detection_csv(file, ids, detections):
    for i, patientId in enumerate(ids):
        dets = detections[i]
        dets = [to_point_form(det) for det in dets]

        line = '{},'.format(patientId)

        for j, det in enumerate(dets):
            if j == 0:
                line = '{}{} {} {} {} {}'.format(
                    line,
                    det[4],
                    int(det[0]),
                    int(det[1]),
                    int(det[2]),
                    int(det[3]),    
                )
            else:
                line = '{} {} {} {} {} {}'.format(
                    line,
                    det[4],
                    int(det[0]),
                    int(det[1]),
                    int(det[2]),
                    int(det[3]),    
                )

        file.write('{}\n'.format(line))

# convert from bbox (xmin, ymin, xmax, ymax) to point-form (xmin, ymin, width, height)
def to_point_form(bbox):
    p = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], bbox[4]]
    return p
