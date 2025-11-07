import os
from tkinter import *
import tkinter
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc,confusion_matrix



main = tkinter.Tk()
main.title("SOFTWARE DEFECT ESTIMATION USING MACHINE LEARNING ALGORITHMS")
main.geometry("1300x1200")

global filename

global balance_data
global data
global X, Y, X_train, X_test, y_train, y_test
global LR_acc, NB_acc, RFT_acc,SVC_acc,DT_acc


def upload():
    global filename
    text.delete('1.0', END)
    filename = askopenfilename(initialdir="SoftwareDataset")
    pathlabel.config(text=filename)
    text.insert(END, "Dataset loaded\n\n")


def preprocess():

    global filename
    global balance_data
    text.delete('1.0', END)
    balance_data = pd.read_csv(filename)

    text.insert(END, "Information about the dataset\n\n")
    text.insert(END, balance_data.head())
    text.insert(END, "\n\n")

    """## 2. Familiarizing with Data & EDA:
    In this step, few dataframe methods are used to look into the data and its features.
    """

    # Shape of dataframe
    text.insert(END, "Shape of dataframe\n\n")
    text.insert(END, balance_data.shape)
    text.insert(END, "\n\n")
    # Listing the features of the dataset


    text.insert(END, "Listing the features of the dataset\n\n")
    text.insert(END, balance_data.columns)
    text.insert(END, "\n\n")

    # Information about the dataset


    # nunique value in columns


    text.insert(END, "nunique value in columns\n\n")
    text.insert(END, balance_data.nunique())
    text.insert(END, "\n\n")

    defect_true_false = balance_data.groupby('defects')['b'].apply(lambda x: x.count())
    print('False: ', defect_true_false[0])
    print('True: ', defect_true_false[1])

    f, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(balance_data.corr(), annot=True, linewidths=.5, fmt='.2f')
    plt.show()
    balance_data.isnull().sum()

    def evaluation_control(data):
        evaluation = (data.n < 300) & (data.v < 1000) & (data.d < 50) & (data.e < 500000) & (data.t < 5000)
        data['complexityEvaluation'] = pd.DataFrame(evaluation)
        data['complexityEvaluation'] = ['Succesful' if evaluation == True else 'Redesign' for evaluation in
                                        data.complexityEvaluation]

    evaluation_control(balance_data)
    from sklearn import preprocessing

    scale_v = balance_data[['v']]
    scale_b = balance_data[['b']]

    minmax_scaler = preprocessing.MinMaxScaler()

    v_scaled = minmax_scaler.fit_transform(scale_v)
    b_scaled = minmax_scaler.fit_transform(scale_b)

    balance_data['v_ScaledUp'] = pd.DataFrame(v_scaled)
    balance_data['b_ScaledUp'] = pd.DataFrame(b_scaled)
    text.insert(END, "\n")
    text.insert(END, balance_data)
    scaled_data = pd.concat([balance_data.v, balance_data.b, balance_data.v_ScaledUp, balance_data.b_ScaledUp], axis=1)
    text.insert(END, "\n")
    text.insert(END, scaled_data)

    text.insert(END, "\n")
    print(balance_data.info())

    text.insert(END, "Removed non numeric characters from dataset\n\n")
    text.insert(END, "\n")
    text.insert(END, "Dataset preprocessing is completed!.")

def generateModel():
    text.delete('1.0', END)
    global X,Y,X_train, X_test, y_train, y_test
    global balance_data
    """## 4. Splitting the Data:
    The data is split into train & test sets, 80-20 split.
    """
    X = balance_data.iloc[:, :-10].values  # Select related attribute values for selection
    Y = balance_data.complexityEvaluation.values  # Select classification attribute values

    # Splitting the dataset into train and test sets: 80-20 split

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

    text.insert(END, "Train & Test Model Generated\n\n")
    text.insert(END,'The data is split into train & test sets, 80-20 split\n\n')
    text.insert(END, "Total Dataset Size : " + str(len(balance_data)) + "\n")
    text.insert(END, "Split Training Size : " + str(len(X_train)) + "\n")
    text.insert(END, "Split Test Size : " + str(len(X_test)) + "\n")



def plot_confusion_matrix(test_Y, predict_y):
 C = confusion_matrix(test_Y, predict_y)
 A =(((C.T)/(C.sum(axis=1))).T)
 B =(C/C.sum(axis=0))
 plt.figure(figsize=(20,4))
 labels = [1,2]
 cmap=sns.light_palette("blue")
 plt.subplot(1, 3, 1)
 sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Confusion matrix")
 plt.subplot(1, 3, 2)
 sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Precision matrix")
 plt.subplot(1, 3, 3)
 sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Recall matrix")
 plt.show()

def runRFT():
    text.delete('1.0', END)
    global RFT_acc
    global classifier
    global X, Y, X_train, X_test, y_train, y_test
    total = X_train.shape[1];
    text.insert(END, "Total Features : " + str(total) + "\n")
    from sklearn.ensemble import RandomForestClassifier

    # instantiate the model
    forest = RandomForestClassifier(n_estimators=10)

    # fit the model
    forest.fit(X_train, y_train)

    # predicting the target value from the model for the samples
    y_train_forest = forest.predict(X_train)
    y_test_forest = forest.predict(X_test)

    # computing the accuracy, f1_score, Recall, precision of the model performance

    acc_train_forest = metrics.accuracy_score(y_train, y_train_forest)
    acc_test_forest = metrics.accuracy_score(y_test, y_test_forest)
    RFT_acc = acc_test_forest
    # computing the accuracy, f1_score, Recall, precision of the model performance
    text.insert(END, "\n")
    text.insert(END, "Random Forest : Accuracy on training Data: {:.3f}".format(acc_train_forest))
    text.insert(END, "\n")
    text.insert(END, "Random Forest : Accuracy on test Data: {:.3f}".format(acc_test_forest))
    text.insert(END, "\n")

    f1_score_train_forest = metrics.f1_score(y_train, y_train_forest)
    f1_score_test_forest = metrics.f1_score(y_test, y_test_forest)
    text.insert(END, "\n")
    text.insert(END, "Random Forest : f1_score on training Data: {:.3f}".format(f1_score_train_forest))
    text.insert(END, "\n")
    text.insert(END, "Random Forest : f1_score on test Data: {:.3f}".format(f1_score_test_forest))
    text.insert(END, "\n")

    recall_score_train_forest = metrics.recall_score(y_train, y_train_forest)
    recall_score_test_forest = metrics.recall_score(y_test, y_test_forest)
    text.insert(END, "\n")
    text.insert(END, "Random Forest : Recall on training Data: {:.3f}".format(recall_score_train_forest))
    text.insert(END, "\n")
    text.insert(END, "Random Forest : Recall on test Data: {:.3f}".format(recall_score_test_forest))
    text.insert(END, "\n")

    precision_score_train_gbc = metrics.precision_score(y_train, y_train_forest)
    precision_score_test_gbc = metrics.precision_score(y_test, y_test_forest)
    text.insert(END, "\n")
    text.insert(END, "Random Forest : precision on training Data: {:.3f}".format(precision_score_train_gbc))
    text.insert(END, "\n")
    text.insert(END, "Random Forest : precision on test Data: {:.3f}".format(precision_score_test_gbc))
    text.insert(END, "\n")
    # computing the classification report of the model
    text.insert(END, "\n")
    text.insert(END, "Classification Report: \n")
    text.insert(END, metrics.classification_report(y_test, y_test_forest))
    plot_confusion_matrix(y_train, y_train_forest)

    fpr, tpr, thresholds = roc_curve(y_test, y_test_forest)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='lightgreen', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.show()


def runLR():
    text.delete('1.0', END)
    global LR_acc
    global classifier
    global X, Y, X_train, X_test, y_train, y_test
    total = X_train.shape[1];

    text.insert(END, "Total Features : " + str(total) + "\n")
    from sklearn.linear_model import LogisticRegression
    # from sklearn.pipeline import Pipeline

    # instantiate the model
    log = LogisticRegression()

    # fit the model
    log.fit(X_train, y_train)

    # predicting the target value from the model for the samples

    y_train_log = log.predict(X_train)
    y_test_log = log.predict(X_test)

    # computing the accuracy, f1_score, Recall, precision of the model performance

    acc_train_svc = metrics.accuracy_score(y_train, y_train_log)
    acc_test_svc = metrics.accuracy_score(y_test, y_test_log)
    LR_acc=acc_test_svc
    # computing the accuracy, f1_score, Recall, precision of the model performance
    text.insert(END, "\n")
    text.insert(END, "Logistic Regression : Accuracy on training Data: {:.3f}".format(acc_train_svc))
    text.insert(END, "\n")
    text.insert(END, "Logistic Regression : Accuracy on test Data: {:.3f}".format(acc_test_svc))
    text.insert(END, "\n")

    f1_score_train_svc = metrics.f1_score(y_train, y_train_log)
    f1_score_test_svc = metrics.f1_score(y_test, y_test_log)
    text.insert(END, "\n")
    text.insert(END, "Logistic Regression : f1_score on training Data: {:.3f}".format(f1_score_train_svc))
    text.insert(END, "\n")
    text.insert(END, "Logistic Regression : f1_score on test Data: {:.3f}".format(f1_score_test_svc))
    text.insert(END, "\n")

    recall_score_train_gbc = metrics.recall_score(y_train, y_train_log)
    recall_score_test_gbc = metrics.recall_score(y_test, y_test_log)
    text.insert(END, "\n")
    text.insert(END, "Logistic Regression : Recall on training Data: {:.3f}".format(recall_score_train_gbc))
    text.insert(END, "\n")
    text.insert(END, "Logistic Regression : Recall on test Data: {:.3f}".format(recall_score_test_gbc))
    text.insert(END, "\n")

    precision_score_train_gbc = metrics.precision_score(y_train, y_train_log)
    precision_score_test_gbc = metrics.precision_score(y_test, y_test_log)
    text.insert(END, "\n")
    text.insert(END, "Logistic Regression : precision on training Data: {:.3f}".format(precision_score_train_gbc))
    text.insert(END, "\n")
    text.insert(END, "Logistic Regression : precision on test Data: {:.3f}".format(precision_score_test_gbc))
    text.insert(END, "\n")
    # computing the classification report of the model
    text.insert(END, "\n")
    text.insert(END, "Classification Report: \n")
    text.insert(END, metrics.classification_report(y_test, y_test_log))
    plot_confusion_matrix(y_train, y_train_log)

    fpr, tpr, thresholds = roc_curve(y_test, y_test_log)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='lightgreen', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.show()


def runsvm():
    text.delete('1.0', END)
    global SVC_acc
    global classifier
    global X, Y, X_train, X_test, y_train, y_test
    total = X_train.shape[1];

    text.insert(END, "Total Features : " + str(total) + "\n")

    from sklearn.ensemble import GradientBoostingClassifier

    # instantiate the model
    gbc = svm.SVC(kernel='rbf', class_weight='balanced', probability=True)

    # fit the model
    gbc.fit(X_train, y_train)
    # predicting the target value from the model for the samples
    y_train_gbc = gbc.predict(X_train)
    y_test_gbc = gbc.predict(X_test)

    # computing the accuracy, f1_score, Recall, precision of the model performance

    acc_train_gbc = metrics.accuracy_score(y_train, y_train_gbc)
    acc_test_gbc = metrics.accuracy_score(y_test, y_test_gbc)
    SVC_acc=acc_test_gbc
    text.insert(END, "\n")
    text.insert(END,"Support Vector Classifier : Accuracy on training Data: {:.3f}".format(acc_train_gbc))
    text.insert(END, "\n")
    text.insert(END,"Support Vector Classifier : Accuracy on test Data: {:.3f}".format(acc_test_gbc))
    text.insert(END, "\n")

    f1_score_train_gbc = metrics.f1_score(y_train, y_train_gbc)
    f1_score_test_gbc = metrics.f1_score(y_test, y_test_gbc)
    text.insert(END, "\n")
    text.insert(END,"Support Vector Classifier : f1_score on training Data: {:.3f}".format(f1_score_train_gbc))
    text.insert(END, "\n")
    text.insert(END,"Support Vector Classifier : f1_score on test Data: {:.3f}".format(f1_score_test_gbc))
    text.insert(END, "\n")

    recall_score_train_gbc = metrics.recall_score(y_train, y_train_gbc)
    recall_score_test_gbc = metrics.recall_score(y_test, y_test_gbc)
    text.insert(END, "\n")
    text.insert(END,"Support Vector Classifier : Recall on training Data: {:.3f}".format(recall_score_train_gbc))
    text.insert(END, "\n")
    text.insert(END,"Support Vector Classifier : Recall on test Data: {:.3f}".format(recall_score_test_gbc))
    text.insert(END, "\n")

    precision_score_train_gbc = metrics.precision_score(y_train, y_train_gbc)
    precision_score_test_gbc = metrics.precision_score(y_test, y_test_gbc)
    text.insert(END, "\n")
    text.insert(END,"Support Vector Classifier : precision on training Data: {:.3f}".format(precision_score_train_gbc))
    text.insert(END, "\n")
    text.insert(END,"Support Vector Classifier : precision on test Data: {:.3f}".format(precision_score_test_gbc))
    text.insert(END, "\n")
    # computing the classification report of the model
    text.insert(END, "\n")
    text.insert(END, "Classification Report: \n")
    text.insert(END,metrics.classification_report(y_test, y_test_gbc))
    plot_confusion_matrix(y_train, y_train_gbc)



    fpr, tpr, thresholds = roc_curve(y_test, y_test_gbc)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='lightgreen', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.show()


def runDT():
    text.delete('1.0', END)
    global DT_acc
    global classifier
    global X, Y, X_train, X_test, y_train, y_test
    total = X_train.shape[1];

    text.insert(END, "Total Features : " + str(total) + "\n")

    from sklearn.tree import DecisionTreeClassifier

    # instantiate the model
    tree =DecisionTreeClassifier()

    # fit the model
    tree.fit(X_train, y_train)

    # predicting the target value from the model for the samples

    y_train_tree = tree.predict(X_train)
    y_test_tree = tree.predict(X_test)

    # computing the accuracy, f1_score, Recall, precision of the model performance

    acc_train_forest = metrics.accuracy_score(y_train, y_train_tree)
    acc_test_forest = metrics.accuracy_score(y_test, y_test_tree)
    DT_acc=acc_test_forest
    # computing the accuracy, f1_score, Recall, precision of the model performance
    text.insert(END, "\n")
    text.insert(END, "Decision Tree : Accuracy on training Data: {:.3f}".format(acc_train_forest))
    text.insert(END, "\n")
    text.insert(END, "Decision Tree : Accuracy on test Data: {:.3f}".format(acc_test_forest))
    text.insert(END, "\n")

    f1_score_train_forest = metrics.f1_score(y_train, y_train_tree)
    f1_score_test_forest = metrics.f1_score(y_test, y_test_tree)
    text.insert(END, "\n")
    text.insert(END, "Decision Tree : f1_score on training Data: {:.3f}".format(f1_score_train_forest))
    text.insert(END, "\n")
    text.insert(END, "Decision Tree : f1_score on test Data: {:.3f}".format(f1_score_test_forest))
    text.insert(END, "\n")

    recall_score_train_forest = metrics.recall_score(y_train, y_train_tree)
    recall_score_test_forest = metrics.recall_score(y_test, y_test_tree)
    text.insert(END, "\n")
    text.insert(END, "Decision Tree : Recall on training Data: {:.3f}".format(recall_score_train_forest))
    text.insert(END, "\n")
    text.insert(END, "Decision Tree : Recall on test Data: {:.3f}".format(recall_score_test_forest))
    text.insert(END, "\n")

    precision_score_train_gbc = metrics.precision_score(y_train, y_train_tree)
    precision_score_test_gbc = metrics.precision_score(y_test, y_test_tree)
    text.insert(END, "\n")
    text.insert(END, "Decision Tree : precision on training Data: {:.3f}".format(precision_score_train_gbc))
    text.insert(END, "\n")
    text.insert(END, "Decision Tree : precision on test Data: {:.3f}".format(precision_score_test_gbc))
    text.insert(END, "\n")
    # computing the classification report of the model
    text.insert(END, "\n")
    text.insert(END, "Classification Report: \n")
    text.insert(END, metrics.classification_report(y_test, y_test_tree))
    plot_confusion_matrix(y_train, y_train_tree)

    fpr, tpr, thresholds = roc_curve(y_test, y_test_tree)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='lightgreen', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.show()
    import joblib
    # Save the model as a pickle in a file
    joblib.dump(model, 'models/DT.pkl')

def runNB():
    text.delete('1.0', END)
    global NB_acc
    global classifier
    global X, Y, X_train, X_test, y_train, y_test
    total = X_train.shape[1];

    text.insert(END, "Total Features : " + str(total) + "\n")

    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import Pipeline

    # instantiate the model
    nb = GaussianNB()

    # fit the model
    nb.fit(X_train, y_train)

    # predicting the target value from the model for the samples
    y_train_nb = nb.predict(X_train)
    y_test_nb = nb.predict(X_test)

    # computing the accuracy, f1_score, Recall, precision of the model performance

    acc_train_forest = metrics.accuracy_score(y_train, y_train_nb)
    acc_test_forest = metrics.accuracy_score(y_test, y_test_nb)
    NB_acc=acc_test_forest
    # computing the accuracy, f1_score, Recall, precision of the model performance
    text.insert(END, "\n")
    text.insert(END, "Naive Bayes Classifier : Accuracy on training Data: {:.3f}".format(acc_train_forest))
    text.insert(END, "\n")
    text.insert(END, "Naive Bayes Classifier : Accuracy on test Data: {:.3f}".format(acc_test_forest))
    text.insert(END, "\n")

    f1_score_train_forest = metrics.f1_score(y_train, y_train_nb)
    f1_score_test_forest = metrics.f1_score(y_test, y_test_nb)
    text.insert(END, "\n")
    text.insert(END, "Naive Bayes Classifier : f1_score on training Data: {:.3f}".format(f1_score_train_forest))
    text.insert(END, "\n")
    text.insert(END, "Naive Bayes Classifier : f1_score on test Data: {:.3f}".format(f1_score_test_forest))
    text.insert(END, "\n")

    recall_score_train_forest = metrics.recall_score(y_train, y_train_nb)
    recall_score_test_forest = metrics.recall_score(y_test, y_test_nb)
    text.insert(END, "\n")
    text.insert(END, "Naive Bayes Classifier : Recall on training Data: {:.3f}".format(recall_score_train_forest))
    text.insert(END, "\n")
    text.insert(END, "Naive Bayes Classifier : Recall on test Data: {:.3f}".format(recall_score_test_forest))
    text.insert(END, "\n")

    precision_score_train_gbc = metrics.precision_score(y_train, y_train_nb)
    precision_score_test_gbc = metrics.precision_score(y_test, y_test_nb)
    text.insert(END, "\n")
    text.insert(END, "Naive Bayes Classifier : precision on training Data: {:.3f}".format(precision_score_train_gbc))
    text.insert(END, "\n")
    text.insert(END, "Naive Bayes Classifier : precision on test Data: {:.3f}".format(precision_score_test_gbc))
    text.insert(END, "\n")
    # computing the classification report of the model
    text.insert(END, "\n")
    text.insert(END, "Classification Report: \n")
    text.insert(END, metrics.classification_report(y_test, y_test_nb))
    plot_confusion_matrix(y_train, y_train_nb)

    fpr, tpr, thresholds = roc_curve(y_test, y_test_nb)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='lightgreen', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.show()


def detectAttack():
    text.delete('1.0', END)
    # then make a url variable
    url = "http://127.0.0.1:5000"
    # then call the get method to select the code
    # for new browser and call open method
    # described above
    import webbrowser
    webbrowser.open(url, new=0, autoraise=True)
    os.system('python app.py')



def graph():
    height = [LR_acc, NB_acc, RFT_acc,SVC_acc,DT_acc]
    bars = ('LR Accuracy', 'NB Accuracy', 'RFT Accuracy','SVC Accuracy','DT Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Explainable AI (XAI) Techniques")
    plt.ylabel("Accuracy Score")
    plt.title("Comparison of Performance Estimation")
    plt.show()


font = ('times', 16, 'bold')
title = Label(main,
              text='SOFTWARE DEFECT ESTIMATION USING MACHINE LEARNING ALGORITHMS')
title.config(bg='pink', fg='purple')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=0)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Software Dataset", command=upload)
upload.place(x=700, y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')
pathlabel.config(font=font1)
pathlabel.place(x=700, y=150)

preprocess = Button(main, text="Dataset Pre-processing", command=preprocess)
preprocess.place(x=700, y=200)
preprocess.config(font=font1)

model = Button(main, text="Feature Classification", command=generateModel)
model.place(x=700, y=250)
model.config(font=font1)


runsvm = Button(main, text="Support Vector Machine Algorithm", command=runsvm)
runsvm.place(x=700, y=300)
runsvm.config(font=font1)

runlr = Button(main, text="Logistics Regression Algorithm", command=runLR)
runlr.place(x=700, y=350)
runlr.config(font=font1)

rundet = Button(main, text="Decision Tree Classifier Algorithm", command=runDT)
rundet.place(x=700, y=400)
rundet.config(font=font1)

runrf = Button(main, text="Random Forest Algorithm", command=runRFT)
runrf.place(x=700, y=450)
runrf.config(font=font1)

runNaB = Button(main, text="Naive Bayes Classifier Algorithm", command=runNB)
runNaB.place(x=700, y=500)
runNaB.config(font=font1)

defectButton = Button(main, text="Software Defect Prediction", command=detectAttack)
defectButton.place(x=700, y=600)
defectButton.config(font=font1)

graphButton = Button(main, text="Comparison of Models", command=graph)
graphButton.place(x=700, y=550)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=80)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=100)
text.config(font=font1)

main.config(bg='purple')
main.mainloop()
