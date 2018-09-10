def export_classification_csv(file, ids, labels, classes, confs):
    for i in range(len(ids)):
        file.write('{},{},{},{}\n'.format(ids[i], labels[i], classes[i], confs[i]))
