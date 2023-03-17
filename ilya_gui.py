import sys
import os
import numpy as np
import keras as ks
import time

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QHBoxLayout  # layout of GUI
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton         # objects of GUI
from PyQt5 import QtCore, QtGui  # variables for GUI

# convert strings to output labels
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences  # pads lists to same size

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pickle

import networkx as nx
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def word_to_nums(user_input, min_symbol=32):
    """
    turns the word into a list of numbers for passing to the input
    """
    return [ord(c) - min_symbol + 1 for c in user_input]


def string_to_numeric(string_to_convert):
    """
    __ Parameters __
    [str] string_to_convert: string to represent as an integer list

    __ Description __
    converts a string to it's numeric list representation. handles a-z and !#$...etc

    __ Return __
    [int-list] numeric_representation: of string
    """

    # 1 - to being enumerating from zero, remove the offest for the first allowed character
    offset = 32

    # 2 - cast char to integer
    numeric_representation = np.array(
        [ord(char) for char in string_to_convert])
    numeric_representation = numeric_representation - offset

    return numeric_representation


def predict(tag, model, vocabulary_size, max_tag_length, output_labels):
    """
    __ Parameters __
    [str] word: word to predict
    [keras] model: model to use to predict word
    [int] vocabulary_size: vocabularly size used to encode the tags during training

    __ Description __
    returns prediction array for the given tag

    __ Return __
    [int] output_numeric, [str] output_string, [arr] probabilities
    """

    # 1 - encode the tag as an integer
    tag_integer = string_to_numeric(tag)

    # 2 - pad to "max_tag_length "and wrap into a a list
    tag_integer = pad_sequences([tag_integer], max_tag_length)

    # 3 - run prediction
    raw_prediction = model.predict(tag_integer)

    # 4 - create array of the form [0, 1, 0 ,0] to indicate highest value
    output_numeric = np.argmax(raw_prediction[0])
    output_string = output_labels[output_numeric]

    return output_numeric, output_string, raw_prediction[0]


def save_parameters(lb, vocabulary_size, max_input_length,
                    parameter_file="./ilya_model/model_param"):
    """
    __ Parameters __
    [LabelBinarizer] lb: sklearn.preprocessing.LabelBinarizer that was used to create unique binaries from list of strings (output)
    [int] vocabulary_size: vocabularly size used to encode input during training (input)
    [int] max_input_length: length of the input (input)
    [str] parameter_file: location to save data

    __ Description __
    stores all the parameters used by the model
    """

    # 1 - extract all the binary labels
    output_map = lb.classes_

    # 2 - dump to file
    with open("./ilya_model/model_param", "wb") as fout:
        pickle.dump({"Output_Map": output_map,
                     "Vocabulary_Size": vocabulary_size,
                     "Max_Input_Length": max_input_length},
                    fout)


def load_parameters(parameter_file="./ilya_model/model_param"):
    """
    __ Parameters __
    [str] parameter_file: file to load from

    __ Description __
    loads parameters to work with model

    __ Return __
    [list-str] output: ordered list of string output ["description", "label"]
    """
    with open("./ilya_model/model_param", "rb") as fin:
        data = pickle.load(fin)

    vocabulary_size = data["Vocabulary_Size"]
    max_input_length = data["Max_Input_Length"]
    output_labels = data["Output_Map"]

    return vocabulary_size, max_input_length, output_labels


def interpret_prediction(p, all_fields):
    """
    interprets the prediction and outputs the field
    """
    return all_fields[np.argmax(p)]


class NNVisualization(QWidget):
    def __init__(self, model_name, model_parameters,
                 train_in, train_out, tags, field):
        """
        __ Parameters __
        [keras.Model] model

        __ Description __
        Widget to visualize neural network
        """

        super().__init__()

        # 1 - load model, parameters and training files
        self.model = ks.models.load_model(model_name)
        self.vocabulary_size, self.max_input_length, self.output_labels = load_parameters(
            param_name)
        self.train_in = np.load(train_in)
        self.train_out = np.load(train_out)
        self.tags = tags
        self.fields = field

        # 2 - set class variables
        self.stop_learning = False
        self.training_step_batch = 200

        # 3 - set title and geometry
        self.setWindowTitle("Tag Predicting NN visualization")
        self.setGeometry(0, 0, 600, 300)

        # 4 - add elements to GUI
        # a - add the canvas for plotting the outputs
        self.canvas = OutputCanvas(
            self.model, all_fields, self.vocabulary_size, self.max_input_length, self.output_labels)

        # b - textbox for user input
        self.user_input_label = QLabel(self)
        self.user_input_label.setText("Tag: ")
        self.user_input = QLineEdit(self)
        self.user_input.textChanged.connect(self.update_plot)
        self.user_input.setMaximumWidth(100)

        # c - button for learning
        self.button = QPushButton("Learn")
        self.button.pressed.connect(self.learn)

        # d - stop learning button
        self.stop_button = QPushButton("Stop")
        self.stop_button.pressed.connect(self.stop_pressed)

        # e - label stating the performance on the test
        self.last_label = QLabel(self)
        self.last_label.setText("")
        self.last_label.setAlignment(QtCore.Qt.AlignCenter)
        self.last_label.setFont(QtGui.QFont("SansSerif", 20))

        # 5 - position of objects
        layout = QVBoxLayout()
        hlayout = QHBoxLayout()
        # a - top row layout
        hlayout.addStretch()
        hlayout.addWidget(self.user_input_label)
        hlayout.addWidget(self.user_input)
        hlayout.addWidget(self.button)
        hlayout.addWidget(self.stop_button)
        hlayout.addStretch()
        # b - vertical layout
        layout.addLayout(hlayout)
        layout.addWidget(self.canvas)
        layout.addWidget(self.last_label)
        # c - full layout
        self.setLayout(layout)

    def update_plot(self):
        """
        __ Description __
        Runs the plot function on the Canvas
        """

        # 1 - grab user input
        user_input = self.user_input.text()
        if user_input == "":
            user_input = None

        # 2 - evaluate user_input and plot on canvas
        self.canvas.plot(user_input)

    def stop_pressed(self):
        """
        __ Description __
        Stops learning function
        """
        self.stop_learning = True

    def learn(self):
        self.stop_learning = False

        # 1 - repeat until stop button is pressed
        while (not self.stop_learning):
            # 2 - do a learning step
            self.learn_step()

            # 3 - put in a random word and show the result
            tag = self.tags[np.random.randint(len(self.tags))]
            self.user_input.setText(tag)
            self.update_plot()

    def learn_step(self):
        """
        __ Description __
        Perform a learning step with the training data
        """

        # 1 - train on 200 random data pieces, iterating over them once (1 epoch)
        train_indx = np.random.randint(0, len(self.train_in), size=200)

        self.model.fit(self.train_in[train_indx, :],
                       self.train_out[train_indx, :],
                       epochs=1, batch_size=32)

        # 2 - pick random 100 samples and see how many you get correct
        correct = 0
        for i in range(100):
            # a - take random test_tag and run prediction
            indx = np.random.randint(len(self.tags))
            test_tag = self.tags[indx]
            test_field = self.fields[indx]

            predict_numeric, predict_string, predict_raw = predict(
                test_tag, self.model,
                self.vocabulary_size, self.max_input_length, self.output_labels)

            if (predict_string == test_field):
                correct += 1

        # 3 - show reults of test
        self.last_label.setText(
            "Learning step done. %i correct out of last 100" % (correct))

        # 4 - update any graphs with the imporved predictions
        self.update_plot()
        time.sleep(0.5)
        app.processEvents(QtCore.QEventLoop.AllEvents, 500)


class OutputCanvas(FigureCanvas):
    def __init__(self, model, all_fields, vocabulary_size, max_input_length, output_labels,
                 parent=None, width=5, height=4, dpi=100):
        """
        __ Parameters __
        [keras.Model] model: to use for running prediction
        [list-string]: all_fields: fields that are are being predicted
        [int] vocabulary_size: vocabularly size used to encode the tags during training
        [int] max_input_length: the tag length the model has been trained to
        [list-str] output_labels: string labels of possible outputs
        ----------------------------------------
        [PyQt5] parent: parent object
        [int] width, height, dpi

        __ Description __
        Performs plotting of neural network progress and output
        """

        # 1 - set class variables
        self.model = model
        self.vocabulary_size = vocabulary_size
        self.max_input_length = max_input_length
        self.output_labels = output_labels
        self.cmap = plt.get_cmap("plasma")

        # 2 - create two axes to plot on
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax_networkx = self.fig.add_subplot(121)
        self.ax_output = self.fig.add_subplot(122)

        # 3 - initialize canvas that will be embeded in the GUI
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)  # parent object
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.plot()

    def plot(self, user_input=None):
        """
        __ Parameters __
        [str] user_input: input to decode

        __ Description __
        plots the prediction and processing on the two charts
        """

        # 1 - clear all graphs on canvas
        self.ax_networkx.clear()
        self.ax_output.clear()

        # 2 - visualize processing and output
        self.plot_networkx(user_input)
        self.plot_prediction(user_input)

        # 3 - update the canvas with new plots
        self.draw()

    def plot_prediction(self, user_input=None):
        """
        __ Parameters __
        [str] use_input: user input to predict

        __ Description __
        Uses the classes neural network to predict the output of a user_input
        """

        if (user_input is None):
            # 1 - if input is empty, plot an empty graph
            prediction = [0] * len(self.output_labels)
            self.ax_output.barh(self.output_labels, prediction)

            self.ax_output.set_title("Waiting on user input")
        else:
            # 2 - run the model and plot a bar chart
            field_numeric, field_string, raw_prediction = predict(user_input,
                                                                  self.model, self.vocabulary_size,
                                                                  self.max_input_length,
                                                                  self.output_labels)

            self.ax_output.barh(self.output_labels, raw_prediction,
                                color=self.cmap(raw_prediction))

            # 3 - get predicted field
            self.ax_output.set_title(
                "Predicted field: %s" % (field_string))

        self.ax_output.set_xlim([0, 1])

    def plot_networkx(self, user_input=None):
        if (user_input is None):
            user_input = ''

        # tag_out, tag_values = predict(user_input, self.all_fields)
        field_numeric, field_string, raw_prediction = predict(user_input,
                                                              self.model, self.vocabulary_size,
                                                              self.max_input_length,
                                                              self.output_labels)

        # 1 - get the number of nodes at each layer
        # a - 15 integers representing word
        n_in = int(self.model.layers[0].input.shape[1])
        # b - portion of nodes in hidden layers
        n_hidden1 = int(int(self.model.layers[4].output.shape[1]) / 2)
        n_hidden2 = n_hidden1
        # c - nodes at output
        n_out = int(self.model.layers[-1].output.shape[1])
        total_nodes = n_in + n_out + n_hidden1 + n_hidden2

        # 2 - populate graph
        G = nx.Graph()
        # a - add the "dots"
        G.add_nodes_from(range(total_nodes))
        # b - connect input to 1st hidden layer
        G.add_edges_from([(i, j) for i in range(n_in)
                          for j in range(n_in, n_in + n_hidden1)])
        # c - connect 1st hidden layer to 2nd hidden layer
        G.add_edges_from([(i, j) for i in range(n_in, n_in + n_hidden1)
                          for j in range(n_in + n_hidden1, n_hidden2 + n_hidden1 + n_in)])
        # d - connect 2nd hidden layer with output
        G.add_edges_from([(i, j) for i in range(n_in + n_hidden1, n_hidden2 + n_hidden1 + n_in)
                          for j in range(n_hidden2 + n_hidden1 + n_in, total_nodes)])

        # 3 - position nodes in columns
        pos_in = [[0, n_in - i - 1] for i in range(n_in)]
        pos_hidden1 = [[1, i] for i in range(n_hidden1)]
        pos_hidden2 = [[2, i] for i in range(n_hidden2)]
        pos_out = [[3, 1], [3, 5], [3, 9], [3, 13]]  # aligned with output

        pos = pos_in + pos_hidden1 + pos_hidden2 + pos_out
        G.add_edges_from(pos)

        # 4 - set connection colours
        if user_input == '':
            colors = np.zeros(total_nodes)
            nx.draw(G, with_labels=True, pos=pos, node_color=colors,
                    cmap='twilight', ax=self.ax_networkx)
        else:
            colors = np.concatenate(
                (np.zeros(n_in + n_hidden2 + n_hidden1), raw_prediction))

            # 5 - set labels
            node_labels = [char for char in user_input] + \
                ['-' for i in range(n_in - len(user_input))] + \
                ["."] * (n_hidden1 + n_hidden2 + n_out)
            # ["{:.1f}".format(i) for i in out_reshaped] + \
            # ["{:.1f}".format(i) for i in out]

            # 6 - plot
            nx.draw(G, with_labels=True, pos=pos, node_color=colors,
                    cmap='cool', ax=self.ax_networkx,
                    labels={i: l for i, l in enumerate(node_labels)})


if (__name__ == '__main__'):
    ########################################
    model_name = './ilya_model/tag_predictor_BASE'
    param_name = "./ilya_model/model_param"
    test_data = "./ilya_model/data_verbose"
    train_in = "./ilya_model/data_in.npy"
    train_out = "./ilya_model/data_out.npy"
    ########################################
    # 1 - every GUI must have at least 1 instance of QApplication running
    app = QApplication([])

    # load the data required for interpretation
    with open(test_data, 'rb') as f:
        data = pickle.load(f)
    tags = data["tags"]
    fields = data["field"]
    all_fields = data["all_fields"]

    gui_box = NNVisualization(model_name, param_name,
                              train_in, train_out, tags, fields)
    gui_box.show()

    # run application and catch any errors
    sys.exit(app.exec_())
