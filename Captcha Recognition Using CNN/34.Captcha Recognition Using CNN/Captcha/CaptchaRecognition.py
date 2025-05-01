from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
from keras.models import model_from_json
import pickle
import os
from keras import layers #for building layers of neural net
from keras.models import Model
from keras.models import load_model
from keras import callbacks
import string
import matplotlib.pyplot as plt

main = tkinter.Tk()
main.title("Captcha Recognition Using CNN")
main.geometry("1300x1200")

global filename
global model
global character, nchar
global X, Y


def uploadDataset():
    global filename
    global labels
    labels = []
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    
def preprocessDataset():
    global X, Y
    global character, nchar
    text.delete('1.0', END)
    character = string.ascii_lowercase + "0123456789" 
    nchar = len(character)
    X = np.zeros((1070,50,200,1))
    Y = np.zeros((5,1070,nchar))
    X = np.load("models/X.txt.npy")
    Y = np.load("models/Y.txt.npy")
    text.insert(END,"Total images found in dataset after processing : "+str(X.shape[0])+"\n")
    text.insert(END,"Total characters using to train CNN : "+str(nchar)+"\n")
    text.insert(END,"Characters list : "+str(character))

    test = X[12]
    test = cv2.resize(test,(200,200))
    cv2.imshow("Processed Image",test)
    cv2.waitKey(0)

def createmodel():
    img = layers.Input(shape=(50,200,1)) # Get image as an input of size 50,200,1
    conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img) #50*200
    mp1 = layers.MaxPooling2D(padding='same')(conv1)  # 25*100
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    mp2 = layers.MaxPooling2D(padding='same')(conv2)  # 13*50
    conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)
    bn = layers.BatchNormalization()(conv3) #to improve the stability of model
    mp3 = layers.MaxPooling2D(padding='same')(bn)  # 7*25
    
    flat = layers.Flatten()(mp3) #convert the layer into 1-D

    outs = []
    for _ in range(5): #for 5 letters of captcha
        dens1 = layers.Dense(64, activation='relu')(flat)
        drop = layers.Dropout(0.5)(dens1) #drops 0.5 fraction of nodes
        res = layers.Dense(nchar, activation='sigmoid')(drop)

        outs.append(res) #result of layers
    
    # Compile model and return it
    model = Model(img, outs) #create model
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
    return model    
    
def trainCNN():
    global model
    text.delete('1.0', END)
    global X, Y
    global character, nchar
    X_train, y_train = X[:970], Y[:, :970]
    X_test, y_test = X[970:], Y[:, 970:]
    if os.path.exists('models/model.json'):
        with open('models/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        json_file.close()    
        model.load_weights("models/model_weights.h5")
        model._make_predict_function()
    else:
        model = createmodel()
        hist = model.fit(X, [Y[0], Y[1], Y[2], Y[3], Y[4]], batch_size=32, epochs=25)
        model.save_weights('models/model_weights.h5')
        model_json = model.to_json()
        with open("models/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('models/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])    
    preds = model.evaluate(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]])
    text.insert(END,"CNN Captcha model generated with Loss on testing data : " + str(preds[0]))        
        
def predictCaptcha():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = img / 255.0 
    res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis])) 
    result = np.reshape(res, (5, 36)) 
    k_ind = []
    probs = []
    for i in result:
        k_ind.append(np.argmax(i))
    capt = ''
    for k in k_ind:
        capt += character[k]
    text.insert(END,'Captcha Recognized as : '+capt)    
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Captcha Recognized as : '+capt, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Captcha Recognized as : '+capt, img)
    cv2.waitKey(0)    


def graph():
    f = open('models/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    accuracy = data['dense_4_accuracy']
    loss = data['dense_4_loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('Captcha CNN Accuracy & Loss Graph')
    plt.show()
    
    
    
font = ('times', 15, 'bold')
title = Label(main, text='Captcha Recognition Using CNN',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Captcha Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=50,y=200)
preprocessButton.config(font=font1)

trainButton = Button(main, text="Train Captcha using CNN", command=trainCNN)
trainButton.place(x=50,y=250)
trainButton.config(font=font1)

predictButton = Button(main, text="Predict Captcha from Test Image", command=predictCaptcha)
predictButton.place(x=50,y=300)
predictButton.config(font=font1)

graphButton = Button(main, text="CNN Accuracy & Loss Graph", command=graph)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)


main.config(bg='magenta3')
main.mainloop()
