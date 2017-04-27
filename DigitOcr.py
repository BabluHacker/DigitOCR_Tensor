import sys
import os

from PyQt4.QtGui import *
from PyQt4 import QtGui
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from sklearn import datasets, svm, metrics


model = load_model('/home/mehedi/AI/MAIN/test/ImgClass.model')




total = 0
predicted = 0
notPredicted = 0

def predict_number(filename):
    img=load_img(filename)
    img=img_to_array(img)

    img = img.reshape((1,) + img.shape)
    img = img / 255.0
    pr = model.predict_classes(img, 32, 1)
    digit = get_digit(str(pr[0]))

    label.setText('The predicted number is     :' + digit)

    print pr[0]
    ##confusion matrices
########

    digits = datasets.load_digits()


    images_and_labels = list(zip(digits.images, digits.target))

    n_samples = len(digits.images)

    print "n_sample ---->"
    print n_samples
    data = digits.images.reshape((n_samples, -1))


    classifier = svm.SVC(gamma=0.001)


    classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])


    expected = digits.target[n_samples / 2:]
    predicted = classifier.predict(data[n_samples / 2:])

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    #########


def get_digit(clsNum):
    if clsNum == "0":
        return "E0"
    elif clsNum == "1":
        return "E1"
    elif clsNum == "2":
        return "E2"
    elif clsNum == "3":
        return "E3"
    elif clsNum == "4":
        return "E4"
    elif clsNum == "5":
        return "E5"
    elif clsNum == "6":
        return "E6"
    elif clsNum == "7":
        return "E7"
    elif clsNum == "8":
        return "E8"
    elif clsNum == "9":
        return "E9"
    elif clsNum == "10":
        return "B0"
    elif clsNum == "11":
        return "B1"
    elif clsNum == "12":
        return "B2"
    elif clsNum == "13":
        return "B3"
    elif clsNum == "14":
        return "B4"
    elif clsNum == "15":
        return "B5"
    elif clsNum == "16":
        return "B6"
    elif clsNum == "17":
        return "B7"
    elif clsNum == "18":
        return "B8"
    elif clsNum == "19":
        return "B9"

def getfile():
    fname = QFileDialog.getOpenFileName(None, 'Open file', '\home', "Image files (*.jpg *.gif)")
    filename=str(fname)
    print filename
    le.setPixmap(QPixmap(fname))
    predict_number(filename)

app=QtGui.QApplication(sys.argv)

window=QtGui.QWidget()
window.setGeometry(50,50,700,500)
window.setWindowTitle("Digit Recognizer, OCR, Tensor")
layout = QVBoxLayout()
le=QLabel("give image.jpg")
le.setPixmap(QPixmap(os.getcwd()+"/lo.png"))
layout.addWidget(le)
btn=QPushButton("image")

btn.clicked.connect(getfile)
layout.addWidget(btn)
label=QLabel("Prediction is      : ")

layout.addWidget(label)
window.setLayout(layout)
window.show()
sys.exit(app.exec_())