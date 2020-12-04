#!/usr/bin/env python
# coding: utf-8
import collections
import itertools
import math
import multiprocessing as mp
import os
import time

import PIL.Image
import cv2
import imblearn
import matplotlib.pyplot as plt
import numpy as np
import pandas
# import skimage.io
import sklearn
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import tensorflow_hub as hub


@tf.function
def cohen_kappa_loss(y_true, y_pred, row_label_vec, col_label_vec, weight_mat, eps=1e-6, dtype=tf.float32):
    labels = tf.matmul(y_true, col_label_vec)
    weight = tf.pow(tf.tile(labels, [1, tf.shape(y_true)[1]]) - tf.tile(row_label_vec, [tf.shape(y_true)[0], 1]), 2)
    weight /= tf.cast(tf.pow(tf.shape(y_true)[1] - 1, 2), dtype=dtype)
    numerator = tf.reduce_sum(weight * y_pred)

    denominator = tf.reduce_sum(
        tf.matmul(
            tf.reduce_sum(y_true, axis=0, keepdims=True),
            tf.matmul(weight_mat, tf.transpose(tf.reduce_sum(y_pred, axis=0, keepdims=True)))
        )
    )

    denominator /= tf.cast(tf.shape(y_true)[0], dtype=dtype)

    return tf.math.log(numerator / denominator + eps)


class CohenKappaLoss(tf.keras.losses.Loss):
    def __init__(self,
                 num_classes,
                 name='cohen_kappa_loss',
                 eps=1e-6,
                 dtype=tf.float32):
        super(CohenKappaLoss, self).__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)

        self.num_classes = num_classes
        self.eps = eps
        self.dtype = dtype
        label_vec = tf.range(num_classes, dtype=dtype)
        self.row_label_vec = tf.reshape(label_vec, [1, num_classes])
        self.col_label_vec = tf.reshape(label_vec, [num_classes, 1])
        self.weight_mat = tf.pow(
            tf.tile(self.col_label_vec, [1, num_classes]) - tf.tile(self.row_label_vec, [num_classes, 1]),
            2) / tf.cast(tf.pow(num_classes - 1, 2), dtype=dtype)

    def call(self, y_true, y_pred, sample_weight=None):
        return cohen_kappa_loss(
            tf.cast(y_true, self.dtype), tf.cast(y_pred, self.dtype), self.row_label_vec, self.col_label_vec,
            self.weight_mat, self.eps, self.dtype
        )

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "eps": self.eps
        }
        base_config = super(CohenKappaLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_model(latest_checkpoint, strategy, model_source, model_name, image_size,
              trainable_layer, fc_size, category_labels, loss_function, tfa_metrics):
    with strategy.scope():
        if model_source == 'keras':
            include_top = False

            if model_name == 'VGG16':
                base_model = tf.keras.applications.VGG16(
                    input_shape=(image_size, image_size, 3), include_top=include_top, weights='imagenet')
                print('VGG16')
            elif model_name == 'VGG19':
                base_model = tf.keras.applications.VGG19(
                    input_shape=(image_size, image_size, 3), include_top=include_top, weights='imagenet')
                print('VGG19')
            elif model_name == 'MobileNetV2':
                base_model = tf.keras.applications.MobileNetV2(
                    input_shape=(image_size, image_size, 3), include_top=include_top, weights='imagenet')
                print('MobileNetV2')
            elif model_name == 'InceptionV3':
                base_model = tf.keras.applications.InceptionV3(
                    input_shape=(image_size, image_size, 3), include_top=include_top, weights='imagenet')
                print('InceptionV3')
            elif model_name == 'Xception':
                base_model = tf.keras.applications.Xception(
                    input_shape=(image_size, image_size, 3), include_top=include_top, weights='imagenet')
                print('Xception')
            elif model_name == 'ResNet50':
                base_model = tf.keras.applications.ResNet50(
                    input_shape=(image_size, image_size, 3), include_top=include_top, weights='imagenet')
                print('ResNet50')
            elif model_name == 'DenseNet201':
                base_model = tf.keras.applications.densenet.DenseNet201(
                    input_shape=(image_size, image_size, 3), include_top=include_top, weights='imagenet')
                print('DenseNet201')
            elif model_name == 'NASNetLarge':
                base_model = tf.keras.applications.nasnet.NASNetLarge(
                    input_shape=(image_size, image_size, 3), include_top=include_top, weights='imagenet')
                print('NASNetLarge')
            else:
                print("Wrong model name")
                exit()

            if trainable_layer > 0:
                for layer in base_model.layers[:-trainable_layer]:
                    layer.trainable = False
            else:
                for layer in base_model.layers:
                    layer.trainable = False

        elif model_source == 'tfhub':
            if model_name == 'MobileNetV2':
                base_model_folder = '/mnt/AI/tfhub/7d894117f08a295a627d24c65df048e34e7ac7d4/'
                print('MobileNetV2')
            elif model_name == 'InceptionV3':
                base_model_folder = '/mnt/AI/tfhub/3f675e18714cfa891d083a31557195a0508e560d/'
                print('InceptionV3')
            elif model_name == 'ResNet50':
                base_model_folder = '/mnt/AI/tfhub/5e690529696a1ca5ff36a5e9c7f7255180ef2364/'
                print('ResNet50')
            elif model_name == 'NASNetLarge':
                base_model_folder = '/mnt/AI/tfhub/c57f54b3f7d0ff4ab1eba180075fb0afe4101034/'
                print('NASNetLarge')
            else:
                print("Wrong model name")
                exit()

            if trainable_layer == 0:
                base_model_trainable = False
            else:
                base_model_trainable = True

            base_model = tf.keras.Sequential([
                hub.KerasLayer(hub.load(base_model_folder),
                               trainable=base_model_trainable,
                               input_shape=(image_size, image_size, 3))
            ])

        else:
            print("Wrong model source")
            exit()

        x = base_model.output
        if model_source == 'keras':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        if fc_size > 0:
            x = tf.keras.layers.Dense(fc_size, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.25)(x)
        predictions = tf.keras.layers.Dense(
            len(category_labels), activation=tf.nn.softmax, name='predictions')(x)
        model_created = tf.keras.Model(inputs=base_model.input, outputs=predictions)

        # model_created.summary()

        # for layer in model_created.layers:
        #     print(layer, layer.trainable)

        if loss_function == 'fl':
            loss = tfa.losses.SigmoidFocalCrossEntropy()

        if loss_function == 'ce' or loss_function == 'wce':
            loss = tf.keras.losses.CategoricalCrossentropy()
            if 2 == len(category_labels):
                loss = tf.keras.losses.BinaryCrossentropy()

        if loss_function == 'ck':
            loss = CohenKappaLoss(len(category_labels))

        metrics = [tf.keras.metrics.CategoricalAccuracy()]
        if 2 == len(category_labels):
            metrics = [tf.keras.metrics.BinaryAccuracy()]

        metrics += [tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    # tf.keras.metrics.TruePositives(name='true_positives'),
                    # tf.keras.metrics.FalsePositives(name='false_positives'),
                    # tf.keras.metrics.TrueNegatives(name='true_negatives'),
                    # tf.keras.metrics.FalseNegatives(name='false_negatives'),
                    # tfa.metrics.CohenKappa(num_classes=len(category_labels)),
                    tfa.metrics.F1Score(num_classes=len(category_labels)),
                    tfa.metrics.FBetaScore(num_classes=len(category_labels))]

        if tfa_metrics != 0:
            metrics += [tfa.metrics.CohenKappa(num_classes=len(category_labels))]

        model_created.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=metrics)
        # model_created.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss=loss, metrics=metrics)

        if latest_checkpoint:
            model_created.load_weights(latest_checkpoint)

        return model_created


def compute_roc(y_val, y_pred_raw, category_labels):
    y_val_one_hot = tf.one_hot(y_val, len(category_labels))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(category_labels)):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(
            y_val_one_hot[:, i], y_pred_raw[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    # print(y_val_one_hot[:, 0])
    # print(y_pred_raw[:, 0])
    # print(fpr[0])
    # print(tpr[0])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(
        y_val_one_hot.numpy().ravel(), y_pred_raw.ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(category_labels))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(category_labels)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(category_labels)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def plot_confusion_matrix(cm, class_names, normalcm=True):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure()
    plt.rcParams["font.family"] = "SimHei"
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    if normalcm:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_roc(fpr, tpr, roc_auc, class_names):
    figure = plt.figure()

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':')

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':')

    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], color='aqua',
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="best")
    return figure


def find_target_layer(model):
    # attempt to find the final convolutional layer in the network
    # by looping over the layers of the network in reverse order
    for layer in reversed(model.layers):
        # check to see if the layer has a 4D output
        if len(layer.output_shape) == 4:
            return layer.name
    # otherwise, we could not find a 4D layer so the GradCAM
    # algorithm cannot be applied
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


def log_heat_map(category_labels, x_val, image_size, model, model_source,
                 result_dir, random_state, initial_epoch):
    for l in category_labels:
        hm_count = 0
        for i in range(len(x_val)):
            if l in x_val[i]:
                x = tf.keras.preprocessing.image.load_img(x_val[i], target_size=(image_size, image_size))
                x = tf.keras.preprocessing.image.img_to_array(x)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x, model_source)

                with tf.GradientTape() as tape:
                    last_conv_layer = model.get_layer(find_target_layer(model))
                    iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
                    model_out, conv_out = iterate(x)
                    class_out = model_out[:, np.argmax(model_out[0])]
                    grads = tape.gradient(class_out, conv_out)
                    pooled_grads = K.mean(grads, axis=(0, 1, 2))

                hm_count += 1
                heatmap = tf.multiply(pooled_grads, conv_out)
                heatmap = tf.reduce_mean(heatmap, axis=-1)
                heatmap = tf.reduce_mean(heatmap, axis=0)
                heatmap = np.maximum(heatmap, 0)
                heatmap /= np.max(heatmap)

                img = cv2.imread(x_val[i])
                INTENSITY = 0.5
                heatmap1 = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                heatmap1 = cv2.applyColorMap(np.uint8(255 * heatmap1), cv2.COLORMAP_VIRIDIS)
                hm_image = heatmap1 * INTENSITY + img
                cv2.imwrite(os.path.join(
                    result_dir,
                    '{}_hm_{}_{}_{}_{}.png'.format(random_state, initial_epoch, l, i, img.shape[1])),
                    hm_image)
                # cv2.imwrite(os.path.join(
                #     result_dir,
                #     '{}_hm_{}_{}_{}_{}_h.png'.format(random_state, initial_epoch, l, i, img.shape[1])),
                #     heatmap1)
                cv2.imwrite(os.path.join(
                    result_dir,
                    '{}_hm_{}_{}_{}_{}_i.png'.format(random_state, initial_epoch, l, i, img.shape[1])),
                    img)

                original_image_file = x_val[i].replace('/{}/'.format(image_size), '/origin/')
                img = cv2.cvtColor(load_crop_image(original_image_file, 1024), cv2.COLOR_RGB2BGR)
                INTENSITY = 0.5
                heatmap1 = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                heatmap1 = cv2.applyColorMap(np.uint8(255 * heatmap1), cv2.COLORMAP_VIRIDIS)
                hm_image = heatmap1 * INTENSITY + img
                cv2.imwrite(os.path.join(
                    result_dir,
                    '{}_hm_{}_{}_{}_{}.png'.format(random_state, initial_epoch, l, i, img.shape[1])),
                    hm_image)
                # cv2.imwrite(os.path.join(
                #     result_dir,
                #     '{}_hm_{}_{}_{}_{}_h.png'.format(random_state, initial_epoch, l, i, img.shape[1])),
                #     heatmap1)
                cv2.imwrite(os.path.join(
                    result_dir,
                    '{}_hm_{}_{}_{}_{}_i.png'.format(random_state, initial_epoch, l, i, img.shape[1])),
                    img)

                # cv2.imwrite(os.path.join(
                #     result_dir,
                #     '{}_hm_{}_{}_{}_o.png'.format(random_state, initial_epoch, l, i)),
                #     img)

                if hm_count >= 100:
                    break


def log_classification_report(y_val, y_pred, category_labels, result_dir, random_state, initial_epoch):
    cr = sklearn.metrics.classification_report(
        y_val, y_pred, target_names=category_labels, output_dict=True)
    cr_df = pandas.DataFrame(cr).transpose()
    cr_df.to_csv(os.path.join(
        result_dir, '{}_classification_report_{}.csv'.format(random_state, initial_epoch)))


def log_auc_sensitivity_specificity(fpr, tpr, result_dir, random_state):
    logfile = open(os.path.join(result_dir, 'auc_sensitivity_specificity.csv'), "a+")
    logfile.write("random, auc, sensitivity0, specificity0, sensitivity1, specificity1, sensitivity, specificity\r\n")
    index_0 = [i for i in range(len(tpr[0])) if tpr[0][i] <= 0.9][-1]
    index_1 = [i for i in range(len(tpr[0])) if tpr[0][i] >= 0.9][0]
    sensitivity = np.array([tpr[0][index_0], tpr[0][index_1]])
    specificity = np.array([1 - fpr[0][index_0], 1 - fpr[0][index_1]])
    logfile.write('{}, {}, {}, {}, {}, {}, {}, {}\r\n'.format(
        random_state, sklearn.metrics.auc(fpr[0], tpr[0]),
        sensitivity[0], specificity[0],
        sensitivity[1], specificity[1],
        0.9, np.interp(0.9, sensitivity, specificity)))
    logfile.close()


def log_predict_validate(x_val, y_val, y_pred, y_pred_raw, category_labels, result_dir,
                         random_state, initial_epoch):
    logfile = open(os.path.join(
        result_dir, '{}_predict_validate_{}.csv'.format(random_state, initial_epoch)), "w+")
    logfile.write("IMAGE, y_real, y_predict, {}\r\n".format(
        ", ".join(str(x) for x in category_labels)))
    for i in range(len(x_val)):
        logfile.write("{}, {}, {}, {}\r\n".format(
            x_val[i], y_val[i], y_pred[i], ", ".join(str(x) for x in y_pred_raw[i])))
    logfile.close()


def log_predict_test(model, test_folders, batch_size, category_labels, image_size, model_source,
                     model_name, result_dir, random_state, initial_epoch, folder_or_csv):
    x_test, y_test = load_data(test_folders, folder_or_csv, category_labels, balance=False)
    y_pred_raw = model.predict(
        input_fn(x_test, y_test, category_labels, 0, image_size, model_source, model_name, batch_size=batch_size))
    y_pred = np.argmax(y_pred_raw, axis=1)
    logfile = open(os.path.join(
        result_dir, '{}_predict_test_{}.csv'.format(random_state, initial_epoch)), "w+")
    logfile.write("IMAGE, y_real, y_predict, {}\r\n".format(
        ", ".join(str(x) for x in category_labels)))
    for i in range(len(x_test)):
        logfile.write("{}, {}, {}, {}\r\n".format(
            x_test[i], y_test[i], y_pred[i], ", ".join(str(x) for x in y_pred_raw[i])))
    logfile.close()


# def verify_image(img_file):
#     try:
#         img = skimage.io.imread(img_file)
#     except Exception as e:
#         print(e)
#         return False
#     return True


def load_crop_image_ben_color(filename, image_size, sigmaX=10):
    image = load_crop_image(filename, image_size)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


def load_crop_image(filename, image_size):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (image_size, image_size))
    return image


def process_origin_image(q, iolock):
    while True:
        stuff = q.get()
        if stuff is None:
            break
        origin_filename, save_filename, image_size = stuff
        # if os.path.isfile(origin_filename) and verify_image(origin_filename):
        if os.path.isfile(origin_filename):
            # x = load_crop_image_ben_color(origin_filename, sigmaX=10)
            x = load_crop_image(origin_filename, image_size)
            PIL.Image.fromarray(x).save(save_filename)


def process_origin_image_folder(source_top_folder, target_top_folder, category_labels, image_size):
    for l in category_labels:
        target_folder = os.path.join(target_top_folder, l)
        if not os.path.exists(target_folder):
            source_folder = os.path.join(source_top_folder, l)
            os.makedirs(target_folder)
            q = mp.Queue(maxsize=os.cpu_count())
            iolock = mp.Lock()
            pool = mp.Pool(os.cpu_count(), initializer=process_origin_image, initargs=(q, iolock))
            for f in os.listdir(source_folder):
                portion = os.path.splitext(f)
                stuff = (os.path.join(source_folder, f),
                         os.path.join(target_top_folder, l, portion[0] + '.jpg'),
                         image_size)
                q.put(stuff)  # blocks until q below its max size
            for _ in range(os.cpu_count()):  # tell workers we're done
                q.put(None)
            pool.close()
            pool.join()


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        # image is too dark so that we crop out everything,
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def load_data(top_folders, folder_or_csv, category_labels, balance):
    x = []
    y = []
    if folder_or_csv:
        for top_folder in top_folders:
            df = pandas.read_csv(os.path.join(top_folder, 'image.csv'))
            for index, row in df.iterrows():
                x += [os.path.join(top_folder, 'image', '{}.png'.format(row['x']))]
                if 'y' in df.columns:
                    y += [row['y']]
                else:
                    y += [0]
    else:
        for l in category_labels:
            for top_folder in top_folders:
                folder = os.path.join(top_folder, l)
                for f in os.listdir(folder):
                    file_path = os.path.join(folder, f)
                    if os.path.isfile(file_path):
                        x += [file_path]
                        y += [category_labels.index(l)]
    print('Before balance : data length : {}, label length : {}'.format(len(x), len(y)))
    print(collections.Counter(y))

    if balance == 1:
        x, y = imblearn.over_sampling.RandomOverSampler().fit_sample(
            np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1))
        x = x.reshape(-1)
        y = y.reshape(-1)
    if balance == 2:
        x, y = imblearn.under_sampling.RandomUnderSampler().fit_sample(
            np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1))
        x = x.reshape(-1)
        y = y.reshape(-1)
    print('After balance : data length : {}, label length : {}'.format(len(x), len(y)))
    print(collections.Counter(y))

    return x, y


def input_fn(features, labels, category_labels, augment, image_size, model_source, model_name, shuffle=False,
             batch_size=32):
    # https://stackoverflow.com/questions/46444018/
    # meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
    one_hot_labels = tf.one_hot(labels, len(category_labels))
    dataset = tf.data.Dataset.from_tensor_slices((features,
                                                  one_hot_labels,
                                                  np.repeat(augment, len(features)),
                                                  np.repeat(image_size, len(features)),
                                                  np.repeat(model_source, len(features)),
                                                  np.repeat(model_name, len(features))))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))
    dataset = dataset.map(load_image_label, num_parallel_calls=os.cpu_count())
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def load_image_label(filename, one_hot_label, augment, image_size, model_source, model_name):
    image_bytes = tf.io.read_file(filename)
    image = tf.cast(tf.image.decode_jpeg(image_bytes, channels=3), tf.float32)
    image = preprocess_input(image, model_source)
    image = tf.image.resize(image, size=(image_size, image_size))
    if augment >= 1:
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_flip_left_right(image)
    if augment >= 2:
        image = tf.image.random_brightness(image, max_delta=0.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    return image, one_hot_label


def preprocess_input(image, model_source):
    # tf.keras.applications image in [-1, 1]
    if model_source == 'keras':
        # https://www.tensorflow.org/tutorials/images/transfer_learning
        image = (image / 127.5) - 1
    # TensorFlow Hub image in [0, 1]
    if model_source == 'tfhub':
        # https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub
        image = image / 255
    return image


def runTask(model_name, batch_size_per_replica, EPOCHS, trainable_layer, fc_size, category_labels,
            model_source, balance, augment, loss_function, data_set, tfa_metrics, random_state,
            train_or_predict, folder_or_csv, root_folder, image_size_scale):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    image_size = 224
    if model_name == 'Xception' or model_name == 'InceptionV3':
        image_size = 299
    if model_name == 'NASNetLarge':
        image_size = 331

    image_size = image_size * image_size_scale
    batch_size_per_replica = math.floor(batch_size_per_replica / image_size_scale)

    top_folders = data_set.split('+')
    train_folders = []
    test_folders = []
    for top_folder in top_folders:
        origin_train_folder = os.path.join(root_folder, top_folder, 'origin', 'train')
        origin_test_folder = os.path.join(root_folder, top_folder, 'origin', 'test')
        train_folder = os.path.join(root_folder, top_folder, '{}'.format(image_size), 'train')
        test_folder = os.path.join(root_folder, top_folder, '{}'.format(image_size), 'test')
        train_folders += [train_folder]
        test_folders += [test_folder]
        process_origin_image_folder(origin_train_folder, train_folder, category_labels, image_size)

    result_dir = os.path.join(
        root_folder, 'result', '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            model_source, model_name, len(category_labels), batch_size_per_replica, image_size,
            fc_size, trainable_layer, balance, augment, loss_function,
            "".join(str(x) for x in top_folders)))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    model_dir = os.path.join('{}_{}'.format(result_dir, random_state))

    strategy = tf.distribute.MirroredStrategy()
    batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    start_time = str(int(time.time()))

    checkpoint_dir = os.path.join(model_dir, 'checkpoint')
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    model = get_model(latest_checkpoint, strategy, model_source, model_name, image_size,
                      trainable_layer, fc_size, category_labels, loss_function, tfa_metrics)

    initial_epoch = 0
    if latest_checkpoint:
        initial_epoch = int(latest_checkpoint[-4:])

    data, labels = load_data(train_folders, folder_or_csv, category_labels, balance=balance)
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
        data, labels,
        random_state=random_state,
        test_size=0.1,
        # shuffle=True,
        stratify=labels)

    print('Train : data length : {}, label length : {}'.format(len(x_train), len(y_train)))
    print('Validate : data length : {}, label length : {}'.format(len(x_val), len(y_val)))
    print('Train : ', collections.Counter(y_train))
    print('Validate : ', collections.Counter(y_val))

    # mix_count = 0
    # for mix_index in range(len(y_train)):
    #     if mix_count < 600:
    #         if y_train[mix_index] == 1:
    #             y_train[mix_index] = 0
    #             mix_count += 1
    # print('Train adjusted : ', collections.Counter(y_train))
    #
    # mix_count = 0
    # for mix_index in range(len(y_val)):
    #     if mix_count < 100:
    #         if y_val[mix_index] == 1:
    #             y_val[mix_index] = 0
    #             mix_count += 1
    # print('Validate adjusted : ', collections.Counter(y_val))

    if train_or_predict == 0:
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'ckpt_{epoch:04d}'), save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join(model_dir)),
            tf.keras.callbacks.CSVLogger(
                os.path.join(model_dir, 'train_{}.log'.format(initial_epoch + EPOCHS)))
        ]

        if loss_function == 'wce':
            class_weight = sklearn.utils.class_weight.compute_class_weight(
                'balanced', np.unique(y_train), y_train)
            class_weight_dict = dict(enumerate(class_weight))
            history = model.fit(
                input_fn(x_train, y_train, category_labels, augment, image_size, model_source, model_name, shuffle=True,
                         batch_size=batch_size),
                initial_epoch=initial_epoch,
                epochs=initial_epoch + EPOCHS,
                class_weight=class_weight_dict,
                validation_data=input_fn(x_val, y_val, category_labels, 0, image_size, model_source, model_name,
                                         batch_size=batch_size),
                callbacks=callbacks)
        else:
            history = model.fit(
                input_fn(x_train, y_train, category_labels, augment, image_size, model_source, model_name, shuffle=True,
                         batch_size=batch_size),
                initial_epoch=initial_epoch,
                epochs=initial_epoch + EPOCHS,
                validation_data=input_fn(x_val, y_val, category_labels, 0, image_size, model_source, model_name,
                                         batch_size=batch_size),
                callbacks=callbacks)

        results = model.evaluate(
            input_fn(x_val, y_val, category_labels, 0, image_size, model_source, model_name, batch_size=batch_size))
        logfile = open(os.path.join(model_dir, 'validate_{}.log'.format(initial_epoch + EPOCHS)), "a+")
        logfile.write("epoch")
        logfile.write(", ")
        logfile.write(", ".join(str(x) for x in model.metrics_names))
        logfile.write("\r\n")
        logfile.write('{}'.format(initial_epoch + EPOCHS))
        logfile.write(", ")
        logfile.write(", ".join(str(x) for x in results))
        logfile.close()

        logfile = open(os.path.join(
            result_dir, '{}_x_train_{}.log'.format(random_state, initial_epoch + EPOCHS)), "w+")
        for i in range(len(x_train)):
            logfile.write('{}, {}\r\n'.format(x_train[i], y_train[i]))
        logfile.close()

        logfile = open(os.path.join(
            result_dir, '{}_x_validate_{}.log'.format(random_state, initial_epoch + EPOCHS)), "w+")
        for i in range(len(x_val)):
            logfile.write('{}, {}\r\n'.format(x_val[i], y_val[i]))
        logfile.close()

        # https://github.com/tensorflow/tensorflow/issues/36477
        model.save(os.path.join(model_dir, 'saved_model', start_time))

    if train_or_predict == 1:
        y_pred_raw = model.predict(
            input_fn(x_val, y_val, category_labels, 0, image_size, model_source, model_name, batch_size=batch_size))
        y_pred = np.argmax(y_pred_raw, axis=1)

        cm = sklearn.metrics.confusion_matrix(y_val, y_pred)
        plot_confusion_matrix(cm, class_names=category_labels, normalcm=True)
        plt.savefig(os.path.join(result_dir, '{}_cm0_{}.png'.format(random_state, initial_epoch)))
        plot_confusion_matrix(cm, class_names=category_labels, normalcm=False)
        plt.savefig(os.path.join(result_dir, '{}_cm1_{}.png'.format(random_state, initial_epoch)))
        log_classification_report(y_val, y_pred, category_labels, result_dir, random_state, initial_epoch)

        fpr, tpr, roc_auc = compute_roc(y_val, y_pred_raw, category_labels)
        plot_roc(fpr, tpr, roc_auc, class_names=category_labels)
        plt.savefig(os.path.join(result_dir, '{}_roc_{}.png'.format(random_state, initial_epoch)))

        if len(category_labels) == 2:
            log_auc_sensitivity_specificity(fpr, tpr, result_dir, random_state)

        if model_source == "keras":
            log_heat_map(category_labels, x_val, image_size, model, model_source,
                         result_dir, random_state, initial_epoch)

        log_predict_validate(x_val, y_val, y_pred, y_pred_raw, category_labels, result_dir,
                             random_state, initial_epoch)
        # log_predict_test(model, test_folders, batch_size, category_labels, image_size, model_source,
        #                  model_name, result_dir, random_state, initial_epoch, folder_or_csv)


def loopTask(model_name, batch_size_per_replica, EPOCHS, TRAINABLE_LAYERS, FC_SIZES, category_labels,
             MODEL_SOURCES, BALANCES, AUGMENTS, LOSS_FUNCTIONS, DATA_SETS, tfa_metrics, RANDOM_STATUES,
             train_or_predict, folder_or_csv, root_folder, IMAGE_SIZE_SCALES):
    for trainable_layer in TRAINABLE_LAYERS:
        for fc_size in FC_SIZES:
            for model_source in MODEL_SOURCES:
                for balance in BALANCES:
                    for augment in AUGMENTS:
                        for loss_function in LOSS_FUNCTIONS:
                            for data_set in DATA_SETS:
                                for random_state in RANDOM_STATUES:
                                    for image_size_scale in IMAGE_SIZE_SCALES:
                                        runTask(model_name, batch_size_per_replica, EPOCHS,
                                                trainable_layer, fc_size, category_labels, model_source,
                                                balance, augment, loss_function, data_set,
                                                tfa_metrics, random_state, train_or_predict,
                                                folder_or_csv, root_folder, image_size_scale)
