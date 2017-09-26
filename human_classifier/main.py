import cv2
import glob
import tensorflow as tf
import random
from util import batchimagebuffer
from human_classifier import build_network


class InriaBuffer(batchimagebuffer.BatchImageBuffer):
    def __init__(self, num_labels, target_size=0):
        batchimagebuffer.BatchImageBuffer.__init__(self, num_labels, target_size)
        self.process = {}
        self.process['original'] = self.get_original_image
        self.process['flipped'] = self.get_flipped_image

    def add_path_label(self, path, label, processings=None):
        image = cv2.imread(path)
        image = cv2.resize(image, tuple(self._target_size))
        images = []
        if processings == None:
            processings = ['original']
        for processing in processings:
            images.append(self.process[processing](image))

        for img in images:
            img = img / 255.
            self.add_reshaped_image_to_buffer(path, img, label)

    def add_paths_labels(self, paths, labels):
        for path, label in zip(paths, labels):
            self.add_path_label(path, label)

    def get_original_image(self, image):
        return image

    def get_flipped_image(self, image):
        return cv2.flip(image, 1)

# # Background list
# not_human = []
# not_human.append('background')
#
# # Human
# human = []
# human.append('full_body_bot')
# human.append('full_body_center')
# human.append('full_body_left')
# human.append('full_body_left_bot')
# human.append('full_body_left_top')
# human.append('full_body_right')
# human.append('full_body_right_bot')
# human.append('full_body_right_top')
# human.append('full_body_top')
# human.append('full_body_wide')
#
# human.append('full_body_without_head_bot')
# human.append('full_body_without_head_center')
# human.append('full_body_without_head_left')
# human.append('full_body_without_head_left_bot')
# human.append('full_body_without_head_left_top')
# human.append('full_body_without_head_right')
# human.append('full_body_without_head_right_bot')
# human.append('full_body_without_head_right_top')
# human.append('full_body_without_head_top')
# human.append('full_body_without_head_wide')
#
# human.append('upper_body_above_knee_bot')
# human.append('upper_body_above_knee_center')
# human.append('upper_body_above_knee_left')
# human.append('upper_body_above_knee_left_bot')
# human.append('upper_body_above_knee_left_top')
# human.append('upper_body_above_knee_right')
# human.append('upper_body_above_knee_right_bot')
# human.append('upper_body_above_knee_right_top')
# human.append('upper_body_above_knee_top')
# human.append('upper_body_above_knee_wide')
#
# human.append('upper_body_bot')
# human.append('upper_body_center')
# human.append('upper_body_left')
# human.append('upper_body_left_bot')
# human.append('upper_body_left_top')
# human.append('upper_body_right')
# human.append('upper_body_right_bot')
# human.append('upper_body_right_top')
# human.append('upper_body_top')
# human.append('upper_body_wide')

# human.append('head_bot')
# human.append('head_center')
# human.append('head_left')
# human.append('head_left_bot')
# human.append('head_left_top')
# human.append('head_right')
# human.append('head_right_bot')
# human.append('head_right_top')
# human.append('head_top')
# human.append('head_wide')
#
# human.append('lower_body_bot')
# human.append('lower_body_center')
# human.append('lower_body_left')
# human.append('lower_body_left_bot')
# human.append('lower_body_left_top')
# human.append('lower_body_right')
# human.append('lower_body_right_bot')
# human.append('lower_body_right_top')
# human.append('lower_body_top')
#
# human.append('lower_body_under_shoulder_bot')
# human.append('lower_body_under_shoulder_center')
# human.append('lower_body_under_shoulder_left')
# human.append('lower_body_under_shoulder_left_bot')
# human.append('lower_body_under_shoulder_left_top')
# human.append('lower_body_under_shoulder_right')
# human.append('lower_body_under_shoulder_right_bot')
# human.append('lower_body_under_shoulder_right_top')
# human.append('lower_body_under_shoulder_top')
# human.append('lower_body_under_shoulder_wide')
# human.append('lower_body_wide')


def make_train_buffer(path, refine=False):
    # Background list
    not_human = []
    not_human.append('background')

    # Human
    human = []
    human.append('full_body_bot')
    human.append('full_body_center')
    # human.append('full_body_left')
    # human.append('full_body_left_bot')
    # human.append('full_body_left_top')
    human.append('full_body_right')
    human.append('full_body_right_bot')
    human.append('full_body_right_top')
    human.append('full_body_top')
    human.append('full_body_wide')

    max_num_of_category = {}
    max_num_of_category['background'] = 400 * len(human)

    category_bundle = []
    category_bundle.append(not_human)
    category_bundle.append(human)
    buffer = InriaBuffer(len(category_bundle), [96, 96])

    images_count = {}
    for label, categories in zip(range(9999), category_bundle):
        buffer.add_categories(categories)
        for category in categories:
            images_count[category] = 0
            filelist = glob.glob(path + '/' + category + '/*.*')
            random.shuffle(filelist)
            if (category in max_num_of_category and refine) or category == 'background':
                filelist = filelist[:max_num_of_category[category]]
            for file in filelist:
                buffer.add_path_label(file, label)
                if file.find('right') != -1:
                    buffer.add_path_label(file, label, ['flipped'])
                images_count[category] += 1

    print('Num of images in each classes')
    for label, categories in zip(range(9999), category_bundle):
        for category in categories:
            print('%02d. ' % label, str(category).rjust(30), ' - ', images_count[category])
    return buffer


def make_test_buffer(path, refine=False):
    # Background list
    not_human = []
    not_human.append('background')

    # Human
    human = []
    human.append('full_body_bot')
    human.append('full_body_center')
    human.append('full_body_left')
    human.append('full_body_left_bot')
    human.append('full_body_left_top')
    human.append('full_body_right')
    human.append('full_body_right_bot')
    human.append('full_body_right_top')
    human.append('full_body_top')
    human.append('full_body_wide')

    max_num_of_category = {}
    max_num_of_category['background'] = 180 * len(human)

    category_bundle = []
    category_bundle.append(not_human)
    category_bundle.append(human)
    buffer = InriaBuffer(len(category_bundle), [96, 96])
    images_count = {}
    for label, categories in zip(range(9999), category_bundle):
        buffer.add_categories(categories)
        for category in categories:
            images_count[category] = 0
            filelist = glob.glob(path + '/' + category + '/*.*')
            random.shuffle(filelist)
            if (category in max_num_of_category and refine) or category == 'background':
                filelist = filelist[:max_num_of_category[category]]
            for file in filelist:
                buffer.add_path_label(file, label)
                images_count[category] += 1

    print('Num of images in each classes')
    for label, categories in zip(range(9999), category_bundle):
        for category in categories:
            print('%02d. ' % label, str(category).rjust(30), ' - ', images_count[category])
    return buffer


if __name__ == '__main__':
    random.seed(0)
    print('Human classifier with inria dataset')
    train_buffer = make_train_buffer('/home/yildbs/Data/INRIA/imadeit/train/', True)
    train_buffer.shuffle()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    lenet = build_network.LeNet(sess)
    lenet.set_train_buffer(train_buffer)
    lenet.build()

    restore = False
    if not restore:
        lenet.train()
    else:
        lenet.restore()

    test_buffer = make_test_buffer('/home/yildbs/Data/INRIA/imadeit/test/', True)
    test_buffer.shuffle()
    lenet.set_test_buffer(test_buffer)
    lenet.test(True)
