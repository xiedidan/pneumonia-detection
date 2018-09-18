def export_classification_csv(file, ids, labels, classes, confs):
    for i in range(len(ids)):
        file.write('{},{},{},{}\n'.format(ids[i], labels[i], classes[i], confs[i]))

def export_detection_csv(file, ids, detections):
    for i, patientId in enumerate(ids):
        dets = detections[i]

        for j, det in enumerate(dets):
            file.write('{},{} {} {} {} {}\n'.format(
                patientId,
                det[4],
                det[0],
                det[1],
                det[2],
                det[3],
            ))
            