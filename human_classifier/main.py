import cv2
import glob
import tensorflow as tf
import random
from util import batchimagebuffer
from human_classifier import build_network


class InriaBuffer(batchimagebuffer.BatchImageBuffer):
    def add_path_label(self, path, label):
        image = cv2.imread(path)
        image = cv2.resize(image, tuple(self._target_size))
        image = image / 255.
        self.add_reshaped_image_to_buffer(path, image, label)

    def add_paths_labels(self, paths, labels):
        for path, label in zip(paths, labels):
            self.add_path_label(path, label)

# Background list
not_human = []
not_human.append('background')


# Human
human = []
human.append('full_body')
human.append('full_body_without_head')
# human.append('head')
# human.append('lower_body')
# human.append('lower_body_under_shoulder')
human.append('upper_body')
# human.append('upper_body_above_knee')

category_bundle = []
category_bundle.append(not_human)
category_bundle.append(human)


def make_buffer(path, refine=False):
    max_num_of_category = {}
    max_num_of_category['background'] = 5400*3
    # max_num_of_category['background'] = 3000
    # max_num_of_category['full_body'] = 1000
    # max_num_of_category['full_body_without_head'] = 1000
    # max_num_of_category['head'] = 1000
    # max_num_of_category['lower_body'] = 1000
    # max_num_of_category['lower_body_under_shoulder'] = 1000
    # max_num_of_category['upper_body'] = 1000
    # max_num_of_category['upper_body_above_knee'] = 1000

    images_count = {}
    buffer = InriaBuffer(len(category_bundle), [128, 128])
    for label, categories in zip(range(9999), category_bundle):
        buffer.add_categories(categories)
        for category in categories:
            images_count[category] = 0
            filelist = glob.glob(path+'/'+category+'/*.*')
            random.shuffle(filelist)
            if category in max_num_of_category and refine:
                filelist = filelist[:max_num_of_category[category]]
            for file in filelist:
                buffer.add_path_label(file, label)
                images_count[category] += 1

    print('Num of images in each classes')
    for label, categories in zip(range(9999), category_bundle):
        buffer.add_categories(categories)
        for category in categories:
            print('%02d. '% label, str(category).rjust(30), ' - ', images_count[category])

    return buffer


if __name__ == '__main__':
    random.seed(0)
    print('Human classifier with inria dataset')
    train_buffer = make_buffer('/home/yildbs/Data/INRIA/imadeit/train/', True)
    train_buffer.shuffle()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    lenet = build_network.LeNet(sess)

    restore = False

    lenet.set_train_buffer(train_buffer)
    lenet.build()

    if not restore:
        lenet.train()
    else:
        lenet.restore()

    test_buffer = make_buffer('/home/yildbs/Data/INRIA/imadeit/test/')
    test_buffer.shuffle()
    lenet.set_test_buffer(test_buffer)
    lenet.test()
