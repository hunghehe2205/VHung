import os
import csv

train_root = 'data_feat/ucf_internvl_train_feature/'
test_root = 'data_feat/ucf_internvl_test_feature/'
train_txt = 'data/Anomaly_Train.txt'
test_txt = 'data/Anomaly_Test.txt'


def make_train_list():
    files = list(open(train_txt))
    normal = []
    count = 0

    with open('list/ucf_intern_rgb.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label'])
        for file in files:
            name = file[:-5]  # remove .mp4\n
            label = file.split('/')[0]
            if 'Normal' in label:
                name = 'Normal/' + name.split('/')[1]
            filename = train_root + name + '__0.npy'
            if os.path.exists(filename):
                if 'Normal' in label:
                    filename = filename[:-5]
                    for i in range(10):
                        normal.append(filename + str(i) + '.npy')
                else:
                    filename = filename[:-5]
                    for i in range(10):
                        writer.writerow([filename + str(i) + '.npy', label])
            else:
                count += 1
                print(filename)

        for file in normal:
            writer.writerow([file, 'Normal'])

    print(f"Train list done. Missing: {count}")


def make_test_list():
    files = list(open(test_txt))
    count = 0

    with open('list/ucf_intern_rgbtest.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label'])
        for file in files:
            file = file.strip()
            label = file.split('/')[0]
            name = file[:-4]  # remove .avi/.mp4
            if 'Normal' in label:
                name = 'Normal/' + name.split('/')[1]
            filename = test_root + name + '__5.npy'
            if os.path.exists(filename):
                writer.writerow([filename, label])
            else:
                count += 1
                print(filename)

    print(f"Test list done. Missing: {count}")


if __name__ == '__main__':
    make_train_list()
    make_test_list()
