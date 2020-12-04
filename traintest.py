import unittest

import train


class Test(unittest.TestCase):
    MODEL_NAME = ""
    BATCH_SIZE = 1
    MODEL_SOURCES = [""]
    TRAIN_OR_PREDICT = 0
    root_folder = ""
    DATA_SETS = []
    category_labels = []
    FOLDER_OR_CSV = 0
    EPOCHS = 1
    FC_SIZES = [128, 256, 512, 1024]
    TRAINABLE_LAYERS = [10000]
    BALANCES = [0, 1, 2]
    AUGMENTS = [0, 1, 2]
    LOSS_FUNCTIONS = ["ce", "wce", "fl", "ck"]
    TFA_METRICS = 0
    RANDOM_STATUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    IMAGE_SIZE_SCALES = [1]

    def test_DenseNet201KERASTRAIN(self):
        self.MODEL_NAME = "DenseNet201"
        self.BATCH_SIZE = 16
        self.MODEL_SOURCES = ["keras"]
        self.TRAIN_OR_PREDICT = 0
        self.loopTask()

    def test_DenseNet201KERASPREDICT(self):
        self.MODEL_NAME = "DenseNet201"
        self.BATCH_SIZE = 16
        self.MODEL_SOURCES = ["keras"]
        self.TRAIN_OR_PREDICT = 1
        self.loopTask()

    def test_InceptionV3KERASTRAIN(self):
        self.MODEL_NAME = "InceptionV3"
        self.BATCH_SIZE = 32
        self.MODEL_SOURCES = ["keras"]
        self.TRAIN_OR_PREDICT = 0
        self.loopTask()

    def test_InceptionV3KERASPREDICT(self):
        self.MODEL_NAME = "InceptionV3"
        self.BATCH_SIZE = 32
        self.MODEL_SOURCES = ["keras"]
        self.TRAIN_OR_PREDICT = 1
        self.loopTask()

    def test_InceptionV3TFHUBTRAIN(self):
        self.MODEL_NAME = "InceptionV3"
        self.BATCH_SIZE = 32
        self.MODEL_SOURCES = ["tfhub"]
        self.TRAIN_OR_PREDICT = 0
        self.loopTask()

    def test_InceptionV3TFHUBPREDICT(self):
        self.MODEL_NAME = "InceptionV3"
        self.BATCH_SIZE = 32
        self.MODEL_SOURCES = ["tfhub"]
        self.TRAIN_OR_PREDICT = 1
        self.loopTask()

    def test_NASNetLargeKERASTRAIN(self):
        self.MODEL_NAME = "NASNetLarge"
        self.BATCH_SIZE = 4
        self.MODEL_SOURCES = ["keras"]
        self.TRAIN_OR_PREDICT = 0
        self.loopTask()

    def test_NASNetLargeKERASPREDICT(self):
        self.MODEL_NAME = "NASNetLarge"
        self.BATCH_SIZE = 4
        self.MODEL_SOURCES = ["keras"]
        self.TRAIN_OR_PREDICT = 1
        self.loopTask()

    def test_NASNetLargeTFHUBTRAIN(self):
        self.MODEL_NAME = "NASNetLarge"
        self.BATCH_SIZE = 4
        self.MODEL_SOURCES = ["tfhub"]
        self.TRAIN_OR_PREDICT = 0
        self.loopTask()

    def test_NASNetLargeTFHUBPREDICT(self):
        self.MODEL_NAME = "NASNetLarge"
        self.BATCH_SIZE = 4
        self.MODEL_SOURCES = ["tfhub"]
        self.TRAIN_OR_PREDICT = 1
        self.loopTask()

    def test_ResNet50KERASTRAIN(self):
        self.MODEL_NAME = "ResNet50"
        self.BATCH_SIZE = 16
        self.MODEL_SOURCES = ["keras"]
        self.TRAIN_OR_PREDICT = 0
        self.loopTask()

    def test_ResNet50KERASPREDICT(self):
        self.MODEL_NAME = "ResNet50"
        self.BATCH_SIZE = 16
        self.MODEL_SOURCES = ["keras"]
        self.TRAIN_OR_PREDICT = 1
        self.loopTask()

    def test_ResNet50TFHUBTRAIN(self):
        self.MODEL_NAME = "ResNet50"
        self.BATCH_SIZE = 16
        self.MODEL_SOURCES = ["tfhub"]
        self.TRAIN_OR_PREDICT = 0
        self.loopTask()

    def test_ResNet50TFHUBPREDICT(self):
        self.MODEL_NAME = "ResNet50"
        self.BATCH_SIZE = 16
        self.MODEL_SOURCES = ["tfhub"]
        self.TRAIN_OR_PREDICT = 1
        self.loopTask()

    def test_MobileNetV2KERASTRAIN(self):
        self.MODEL_NAME = "MobileNetV2"
        self.BATCH_SIZE = 32
        self.MODEL_SOURCES = ["keras"]
        self.TRAIN_OR_PREDICT = 0
        self.loopTask()

    def test_MobileNetV2KERASPREDICT(self):
        self.MODEL_NAME = "MobileNetV2"
        self.BATCH_SIZE = 32
        self.MODEL_SOURCES = ["keras"]
        self.TRAIN_OR_PREDICT = 1
        self.loopTask()

    def test_MobileNetV2TFHUBTRAIN(self):
        self.MODEL_NAME = "MobileNetV2"
        self.BATCH_SIZE = 32
        self.MODEL_SOURCES = ["tfhub"]
        self.TRAIN_OR_PREDICT = 0
        self.loopTask()

    def test_MobileNetV2TFHUBPREDICT(self):
        self.MODEL_NAME = "MobileNetV2"
        self.BATCH_SIZE = 32
        self.MODEL_SOURCES = ["tfhub"]
        self.TRAIN_OR_PREDICT = 1
        self.loopTask()

    def test_VGG16KERASTRAIN(self):
        self.MODEL_NAME = "VGG16"
        self.BATCH_SIZE = 32
        self.MODEL_SOURCES = ["keras"]
        self.TRAIN_OR_PREDICT = 0
        self.loopTask()

    def test_VGG16KERASPREDICT(self):
        self.MODEL_NAME = "VGG16"
        self.BATCH_SIZE = 32
        self.MODEL_SOURCES = ["keras"]
        self.TRAIN_OR_PREDICT = 1
        self.loopTask()

    def test_VGG19KERASTRAIN(self):
        self.MODEL_NAME = "VGG19"
        self.BATCH_SIZE = 32
        self.MODEL_SOURCES = ["keras"]
        self.TRAIN_OR_PREDICT = 0
        self.loopTask()

    def test_VGG19KERASPREDICT(self):
        self.MODEL_NAME = "VGG19"
        self.BATCH_SIZE = 32
        self.MODEL_SOURCES = ["keras"]
        self.TRAIN_OR_PREDICT = 1
        self.loopTask()

    def test_XceptionKERASTRAIN(self):
        self.MODEL_NAME = "Xception"
        self.BATCH_SIZE = 16
        self.MODEL_SOURCES = ["keras"]
        self.TRAIN_OR_PREDICT = 0
        self.loopTask()

    def test_XceptionKERASPREDICT(self):
        self.MODEL_NAME = "Xception"
        self.BATCH_SIZE = 16
        self.MODEL_SOURCES = ["keras"]
        self.TRAIN_OR_PREDICT = 1
        self.loopTask()

    def loopTask(self):
        train.loopTask(self.MODEL_NAME, self.BATCH_SIZE, self.EPOCHS, self.TRAINABLE_LAYERS, self.FC_SIZES,
                       self.category_labels, self.MODEL_SOURCES, self.BALANCES, self.AUGMENTS, self.LOSS_FUNCTIONS,
                       self.DATA_SETS, self.TFA_METRICS, self.RANDOM_STATUES, self.TRAIN_OR_PREDICT, self.FOLDER_OR_CSV,
                       self.root_folder, self.IMAGE_SIZE_SCALES)


if __name__ == '__main__':
    unittest.main()
