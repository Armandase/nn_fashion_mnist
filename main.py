import os
import numpy as np

# remove tf debugging mode
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from constants import *
from image_manipulation import *
from NeuralNetwork import *

batch_size = 64
learning_rate = 0.1
epochs = 5


def fashion_dataset(predict=False):
    (training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    train_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    train_dataset = train_dataset.map(lambda image, label: (float(image) / 255.0, label))
    test_dataset = test_dataset.map(lambda image, label: (float(image) / 255.0, label))

    if predict:
        return train_dataset, test_dataset
    # print(train_dataset.as_numpy_iterator().next()[0])
    train_dataset = train_dataset.batch(batch_size).shuffle(500)
    test_dataset = test_dataset.batch(batch_size).shuffle(500)
    return train_dataset, test_dataset


def train_model(train_dataset, test_dataset):
    model = NeuralNetwork()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    metrics = ['accuracy']
    model.compile(optimizer, loss_fn, metrics)

    model.fit(train_dataset, epochs=epochs)

    test_loss, test_acc = model.evaluate(test_dataset)
    print(f'\nTest accuracy: {test_acc * 100:>0.1f}%, test loss: {test_loss:>8f}')

    model.save('outputs/model')


def main():
    if not os.path.isdir('outputs/model'):
        train_dataset, test_dataset = fashion_dataset()
        train_model(train_dataset, test_dataset)
    model = tf.keras.models.load_model('outputs/model')

    train_dataset, test_dataset = fashion_dataset(predict=True)

    pred_idx = random.randint(0, len(test_dataset) - 1)
    predict_img = list(test_dataset.as_numpy_iterator())[pred_idx]

    pred_img_array = predict_img[0]
    pred_label = predict_img[1]
    plot_array(pred_img_array, pred_label)

    img_to_pred = np.asarray(pred_img_array, dtype=np.float32).reshape((-1, 28, 28))
    predicted_vector = model.predict(img_to_pred)
    predicted_label = np.argmax(predicted_vector)
    predicted_name = LABELS_MAP[predicted_label]

    probs = tf.nn.softmax(predicted_vector.reshape((-1,)))
    for i, p in enumerate(probs):
        print(f'{LABELS_MAP[i]} -> {p*100:.3f}%')

    print(f'Predicted class : {predicted_name}')


if __name__ == '__main__':
    main()
