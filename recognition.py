import pdb
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import dlib
import sys
import os
import glob
import numpy as np
import _pickle as cPickle
import argparse
#cnn_face = 'mmod_human_face_detector.dat'
#detector = dlib.cnn_face_detection_model_v1(cnn_face)

predictor = 'shape_predictor_5_face_landmarks.dat'
face_rec = 'dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor)
facerec = dlib.face_recognition_model_v1(face_rec)
THRESH = 0.5
width, height = 800, 600
class CamView():  
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.protocol("WM_DELETE_WINDOW",
        self.close) 
        self.show_frame()

    def show_frame(self):
        imgtk = ImageTk.PhotoImage(image=self.parent.img)
        self.parent.lmain.imgtk = imgtk
        self.parent.lmain.configure(image=imgtk)

    def close(self):
        self.parent.test_frame = None
        self.window.destroy()

class Main(tk.Frame):
    def __init__(self, parent):
        self.count=0
        self.lmain = lmain
        self.test_frame = None
        self.frame = tk.Frame.__init__(self,parent)
        self.name_title = tk.Label(text='Nome:').pack(side=tk.TOP,
            anchor=tk.W)

        self.name_field = tk.Entry(bd = 5)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.name_field.pack(side = tk.TOP,anchor=tk.W)

        self.do_stuff()
        self.load_window()
        b_compareData = tk.Button(parent, 
            text='Compare esta imagem com a base',
            width=20,wraplength=80,command=self.compare_data).pack(side=tk.RIGHT,anchor=tk.E)
        b_insertData = tk.Button(parent, 
            text='Adicionar imagem na base', 
            width=20,command=self.insert_data).pack(side=tk.TOP,anchor=tk.NW)

        b_saveData = tk.Button(parent, 
            text='Salvar base de dados',
            width=20,command=self.save_database).pack(side=tk.TOP,anchor=tk.NW)

        b_saveData = tk.Button(parent, 
            text='Visualisar base de dados',
            width=20,command=self.show_database).pack(side=tk.LEFT)

        


    def do_stuff(self):
        _, self.frame = self.cap.read()
        frame = cv2.flip(self.frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        self.img = Image.fromarray(cv2image)

        if self.test_frame != None:
            self.test_frame.show_frame()
        self.lmain.after(10, self.do_stuff)

    def load_window(self):
        self.test_frame = CamView(self)

    def show_database(self):
        faces = []
        if not gallery:
            print ("Base de dados %s vazia"%data_file)
        else:
            for d in gallery:
                bb = d[2]
                face = d[1][bb[0]:bb[1],bb[2]:bb[3]]
                face = cv2.putText(face,'%s'%d[-1],
                    (30,30),cv2.FONT_HERSHEY_DUPLEX,.7,(200,0,0),1)
                face = cv2.resize(face,(150,150))
                faces.append(face)
            mosaic = np.concatenate(faces,axis=1)
            cv2.imshow('Base de dados',mosaic)
            cv2.waitKey(1000)
    def compare_data(self):
        try:
            self.extract_features()
            conf_list = []
            print('Verificando dados existentes na base de dados...')
            for data in gallery:
                conf = np.linalg.norm(np.asarray(self.desc) - np.asarray(data[0]))
                if (conf <= THRESH):
                    conf_list.append([conf,data[1:]])
                    print('Identidade provavel: %s'%data[-1])
            sorted_res = sorted(zip(conf_list), key=lambda x: x[0])
            self.show_matching(sorted_res[0])
        except:
            pass 

    def insert_data(self):
        self.name = self.name_field.get()
        if not self.name:
            self.name = 'New_person'
        try: 
            self.extract_features()
            print('%s foi adicionado na base de dados'%self.name)
            gallery.append([self.desc,self.frame,self.bb,self.name])
            self.save_image() 
        except:
            pass
    def show_matching(self,sorted_res):
        self.count+=1
        img1 = cv2.resize(self.frame,(int(width/2),int(height/2)))
        img2 = cv2.resize(sorted_res[0][1][0],(int(width/2),int(height/2)))
        newimg = np.concatenate((img1,img2),axis=1)
        cv2.putText(newimg,'Imagem teste',(int(width/5),30),
            cv2.FONT_HERSHEY_DUPLEX,.7,(200,0,0),1)
        cv2.putText(newimg,'Imagem da base de dados',(int(width/2*1.1),30),
            cv2.FONT_HERSHEY_DUPLEX,.7,(200,0,0),1)
        cv2.putText(newimg,'%s'%sorted_res[0][1][-1],(int(width/2*1.2),
               (int(height/2*.9))),cv2.FONT_HERSHEY_DUPLEX,.7,(0,0,200),1)
        cv2.imshow('Resultado %d'%self.count,newimg)
        cv2.imwrite('matching%d.jpg'%self.count,newimg)
        cv2.waitKey(1000)
        
    def save_image(self):
        self.img.save('images/%s.png'%self.name)

    def save_database(self):
        print('Base de dados %s salva no arquivo'%data_file)
        cPickle.dump(gallery,open('%s'%data_file,'wb'))

    def extract_features(self):
        dets = detector(self.frame, 1)
        d = dets[0]
        shape = sp(self.frame, d)
        self.desc = facerec.compute_face_descriptor(self.frame, shape)
        self.bb = [d.top(),d.bottom(),d.left(),d.right()]
#face_crop = img[d.top():d.bottom(),d.left():d.right()]
                
parse = argparse.ArgumentParser()
parse.add_argument('-d',type=str,default='data.pkl')
args = parse.parse_args()

if not args.d:
    data_file = 'data.pkl'
else:
    data_file = args.d

try:
    gallery = cPickle.load(open('%s'%data_file,'rb'))
    print('Carregando base de dados %s '%data_file)
except:
    gallery = []
root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
lmain = tk.Label(root)
lmain.pack()
control = Main(root)
root.mainloop()


