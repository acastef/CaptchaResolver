from faster_rcnn import *
import itertools
from keras.layers import Reshape, Lambda, LSTM, Activation
from keras.layers.merge import add, concatenate
from util.generator import ImageGenerator
from util.generator import LETTERS

def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def predict(img_w, max_text_len, weights):
    # Input Parameters
    img_h = 64

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirecitonal LSTMs
    lstm_1 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)
    lstm_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(
        inner)
    lstm1_merged = add([lstm_1, lstm_1b])
    lstm_2 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    lstm_2b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(
        lstm1_merged)

    # transforms RNN output to character activations:
    inner = Dense(ImageGenerator.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([lstm_2, lstm_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    model.load_weights(weights)
    return model

# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.
def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(LETTERS):
                outstr += LETTERS[c]
        ret.append(outstr)
    return ret


base_path = '../../model/faster_rcnn'

test_base_path = '../../data/captcha_detection/test' # Test images

config_output_filename = os.path.join(base_path, 'model_vgg_config.pickle')

target_path = '../../data/demo'

imgs_path = ['3630.png']

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

num_features = 512

input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (VGG here, can be Resnet50, Inception, etc)
shared_layers = nn_base(img_input)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = rpn_layer(shared_layers, num_anchors)

classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

model_rpn.load_weights('../../model/faster_rcnn/model_frcnn_vgg.hdf5', by_name=True)
model_classifier.load_weights('../../model/faster_rcnn/model_frcnn_vgg.hdf5', by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

# Switch key value for class mapping
class_mapping = C.class_mapping
class_mapping = {v: k for k, v in class_mapping.items()}
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}


all_imgs = []

classes = {}

# If the box classification value is less than this, we ignore this box
bbox_threshold = 0.7

for idx, img_name in enumerate(imgs_path):
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    st = time.time()
    filepath = os.path.join(test_base_path, img_name)

    img = cv2.imread(filepath)

    plt.figure(figsize=(10, 10))
    plt.title('Original image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    print('Detecting CAPTCHAs')

    X, ratio = format_img(img, C)

    X = np.transpose(X, (0, 2, 3, 1))

    # get output layer Y1, Y2 from the RPN and the feature maps F
    # Y1: y_rpn_cls
    # Y2: y_rpn_regr
    [Y1, Y2, F] = model_rpn.predict(X)

    # Get bboxes by applying NMS
    # R.shape = (300, 4)
    R = rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        # Calculate bboxes coordinates on resized image
        for ii in range(P_cls.shape[1]):
            # Ignore 'bg' class
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append(
                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    crops = []
    clone_image = img.copy()
    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]

            # Calculate real coordinates on original image
            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            crops.append((real_x1, real_y1, real_x2, real_y2))

            cv2.rectangle(clone_image, (real_x1, real_y1), (real_x2, real_y2),
                          (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 4)

    print('Elapsed time = {}'.format(time.time() - st))
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(clone_image, cv2.COLOR_BGR2RGB))
    plt.show()

    sess = tf.Session()
    K.set_session(sess)

    tiger_test = ImageGenerator('../../data/demo', 128, 64, 1, 4)
    tiger_test.build_data()

    model = predict(128, tiger_test.max_text_len,
                    '../../model/rnn/rnnLSTM6-7-8.hdf5')

    net_inp = model.get_layer(name='the_input').input
    net_out = model.get_layer(name='softmax').output

    predicted_text = []

    for crop in crops:
        cr = img[crop[1]:crop[3], crop[0]:crop[2]]
        cv2.imwrite(os.path.join(target_path, 'temp.png'), cr)

        tiger_test = ImageGenerator('../../data/demo', 128, 64, 1, 4)
        tiger_test.build_data()

        for inp_value, _ in tiger_test.next_batch():
            bs = inp_value['the_input'].shape[0]
            X_data = inp_value['the_input']
            net_out_value = sess.run(net_out, feed_dict={net_inp: X_data})
            pred_texts = decode_batch(net_out_value)[0]
            labels = inp_value['the_labels']
            texts = []
            for label in labels:
                text = ''.join(list(map(lambda x: LETTERS[int(x)], label)))
                texts.append(text)
            pred_texts = pred_texts.replace('_', '')
            predicted_text.append(pred_texts)
            break
    plt.figure(figsize=(15, 5))
    plt.imshow(cv2.cvtColor(clone_image, cv2.COLOR_BGR2RGB))

    text = ''
    for t in predicted_text:
        text += 'Predicted: %s\n' % t

    plt.text(870, 250, text, fontsize=15)
    plt.show()

