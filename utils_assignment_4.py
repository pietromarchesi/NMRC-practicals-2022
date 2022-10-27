import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import ipywidgets as widgets
from IPython import display


class ffnn_simulation():

    def __init__(self):
        self.batch_size = 128
        self.num_classes = 10

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        self.x_train = x_train
        self.x_test = x_test
        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.num_classes)

        self.l_widget = widgets.BoundedIntText(
            value=1,
            min=1,
            max=10,
            description='Number of layers:',
            disabled=False)

        self.n_widget = widgets.BoundedIntText(
            value=100,
            min=1,
            max=1000,
            description='# units per layer:',
            disabled=False)

        self.ne_widget = widgets.BoundedIntText(
            value=20,
            min=1,
            max=100,
            description='# epochs:',
            disabled=False)


    def train(self, n_units, n_epochs):
        model = Sequential()

        model.add(Dense(n_units[0], activation='relu', input_shape=(784,)))

        if len(n_units) > 1:
            for nu in n_units[1:]:
                model.add(Dense(nu, activation='relu'))

        model.add(Dense(self.num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

        history = model.fit(self.x_train, self.y_train,
                            batch_size=self.batch_size,
                            epochs=n_epochs,
                            verbose=1,
                            validation_data=(self.x_test, self.y_test))
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])



    def start(self):
        display.display(self.l_widget)
        display.display(self.n_widget)
        display.display(self.ne_widget)

        def _train(b):
            n_units = [self.n_widget.value for _ in range(self.l_widget.value)]
            n_epochs = self.ne_widget.value
            self.train(n_units, n_epochs)

        button = widgets.Button(description="Initialize and train")
        display.display(button)
        button.on_click(_train)
        display.clear_output(wait=True)


