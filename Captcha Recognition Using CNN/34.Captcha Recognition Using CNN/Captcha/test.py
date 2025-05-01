import numpy as np
import cv2
import string
import os
import matplotlib.pyplot as plt #for graphs
import os #for operating system dependent fucntionality
from keras import layers #for building layers of neural net
from keras.models import Model
from keras.models import load_model
from keras import callbacks
from keras.models import model_from_json
import pickle
'''
X = np.zeros((1070,50,200,1))

character= string.ascii_lowercase + "0123456789" 
nchar = len(character)
print(nchar)
Y = np.zeros((5,1070,nchar))

for i, directory in enumerate(os.listdir("Dataset")):
    if 'Thumbs.db' not in directory:
        img_name = directory.split(".")
        img_name = img_name[0].lower()
        img = cv2.imread("Dataset/"+directory,cv2.IMREAD_GRAYSCALE)
        img = img / 255.0            
        img = np.reshape(img, (50, 200, 1))
        target = np.zeros((5,nchar))
        for k in range(len(img_name)):
            name = img_name[k]
            index = character.find(name)
            #print(str(k)+" "+name+" "+str(index))
            target[k, index] = 1
        print(directory+" "+img_name+" "+str(target))    
        X[i] = img
        Y[:,i] = target
                
np.save("models/X.txt",X)
np.save("models/Y.txt",Y)
'''
X = np.load("models/X.txt.npy")
Y = np.load("models/Y.txt.npy")
print(X.shape)
print(Y.shape)

character= string.ascii_lowercase + "0123456789" 
nchar = len(character)
print(nchar)
'''
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

'''


with open('models/model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
json_file.close()    
model.load_weights("models/model_weights.h5")
model._make_predict_function()


img = cv2.imread('Dataset/2gyb6.png', cv2.IMREAD_GRAYSCALE)
img = img / 255.0 #Scale image
res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis])) 
result = np.reshape(res, (5, 36)) 
k_ind = []
probs = []
for i in result:
    k_ind.append(np.argmax(i)) #adds the index of the char found in captcha

capt = '' #string to store predicted captcha
for k in k_ind:
    capt += character[k] #finds the char corresponding to the index
print(capt)

