import traintest


class Test(traintest.Test):
    def setUp(self):
        self.root_folder = "/mnt/AI/HM/"
        # self.DATA_SETS = ["izy"]
        # self.category_labels = ['D001豹纹状眼底', 'D002弥漫性萎缩', 'D003斑片状萎缩', 'D004黄斑萎缩', 'D005正常']
        # self.DATA_SETS = ["izy20200531c2"]
        # self.category_labels = ['NPM', 'PM']
        # self.DATA_SETS = ["izy20200531c3"]
        # self.category_labels = ['NPM', 'PM', 'LQ']
        self.DATA_SETS = ["izy20200531c5"]
        self.category_labels = ['0其他', '1豹纹', '2弥漫', '3斑片', '4黄斑']
        self.FOLDER_OR_CSV = 0
        self.EPOCHS = 10
        self.FC_SIZES = [32]
        self.TRAINABLE_LAYERS = [10000]
        self.BALANCES = [0]
        self.AUGMENTS = [0]
        self.LOSS_FUNCTIONS = ["ce", "wce", "fl"]
        self.TFA_METRICS = 0
        self.RANDOM_STATUES = [0]

    def test_ALL(self):
        # self.test_DenseNet201KERASTRAIN()
        # self.test_DenseNet201KERASPREDICT()
        # self.test_InceptionV3KERASTRAIN()
        # self.test_InceptionV3KERASPREDICT()
        # self.test_InceptionV3TFHUBTRAIN()
        # self.test_InceptionV3TFHUBPREDICT()
        # self.test_NASNetLargeKERASTRAIN()
        # self.test_NASNetLargeKERASPREDICT()
        # self.test_NASNetLargeTFHUBTRAIN()
        # self.test_NASNetLargeTFHUBPREDICT()
        # self.test_ResNet50KERASTRAIN()
        # self.test_ResNet50KERASPREDICT()
        # self.test_ResNet50TFHUBTRAIN()
        # self.test_ResNet50TFHUBPREDICT()
        # self.test_MobileNetV2KERASTRAIN()
        # self.test_MobileNetV2KERASPREDICT()
        # self.test_MobileNetV2TFHUBTRAIN()
        # self.test_MobileNetV2TFHUBPREDICT()
        # self.test_VGG16KERASTRAIN()
        # self.test_VGG16KERASPREDICT()
        # self.test_VGG19KERASTRAIN()
        # self.test_VGG19KERASPREDICT()
        self.test_XceptionKERASTRAIN()
        self.test_XceptionKERASPREDICT()
