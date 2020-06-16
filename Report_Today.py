import tkinter
from tkinter import ttk
import os, sys
from tkinter import messagebox
from tkinter import filedialog
from pathlib import Path
import pathlib
from pdf2image import convert_from_path
import numpy as np
import cv2
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import skimage.util
import locale
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import calendar

def filedialog_clicked():
    fTyp = [("", "*")]
    iFile = os.path.abspath(os.path.dirname(__file__))
    iFilePath = filedialog.askopenfilename(filetype = fTyp, initialdir = iFile)
    entry2.set(iFilePath)

def pdf_to_list():
    image_path_list, image_path, name_list = pdf_to_image()
    print(name_list)
    image_path_number = len(image_path_list)
    entry2.set(str(image_path.name))
    file_path = os.path.abspath(str(image_path))
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    file_name = file_name[22:]
    file_day = os.path.basename(file_path)[:8]
    df_today_summary = pd.DataFrame(columns=["品名","加工","数量"])
    image_list_summary = []
    
    for name, image_path in zip(name_list, image_path_list):
        file_path = os.path.abspath(str(image_path))
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_name = file_name[22:]
        file_day = os.path.basename(file_path)[:8]
        
        images = image_to_summary(file_path)
        df_today,image_list = result_cognization(np.array(images),name)
        df_result = result_to_list(df_today,name)
        df_today_summary = pd.concat([df_today_summary,df_result],axis=0)
        image_list_summary.append(image_list)
        
    else:
        error_determine(image_list_summary, df_today_summary, name_list, file_day)
        df_today_summary.to_csv('C:\\Report_Today\\csv_file\\result_today_test.csv', sep=',', encoding='cp932')
        result_list = df_today_summary.values.tolist()
    
    tree.heading("date", text = "日付")
    tree.heading("member_number", text = "名前")
    tree.heading("product", text = "品名")
    tree.heading("work", text = "加工")
    tree.heading("number", text = "数量")
    for i in tree.get_children():
        tree.delete(i)
    for i in range(len(result_list)):
        tree.insert("", "end", text="work", values=result_list[i])
    return df_today_summary


def pdf_to_image():
    poppler_dir = Path("C:\\Report_Today\\poppler").parent.absolute() / "poppler\\bin"
    os.environ["PATH"] += os.pathsep + str(poppler_dir)

    pdf_path = Path(entry2.get())
    image_dir = Path("C:\\Report_Today\\image_file")
    
    image = convert_from_path(str(pdf_path), 200, fmt="JPEG")
    
    image_test = image[0]
    image_test = np.asarray(image_test)
    
    name_list = []
    image_path_list = []
    if select_date_button["text"] == "日付選択":
        now_time = date_get()
        for i, page in enumerate(image):
            page = np.asarray(page)
            page = image_offset(page)
            name = file_name_recognize(page)
            file_name = now_time.strftime('%Y%m%d') + "_Report_Today" + "_{0}".format(name) + ".jpg"
            image_path = image_dir / file_name

            cv2.imwrite(str(image_path), page)
            name_list.append(name)
            image_path_list.append(str(image_path))
    else:
        select_day = select_date_button["text"].replace("/","")
        for i, page in enumerate(image):
            page = np.asarray(page)
            page = image_offset(page)
            name = file_name_recognize(page)
            file_name = select_day + "_Report_Today" + "_{0}".format(name) + ".jpg"
            image_path = image_dir / file_name

            cv2.imwrite(str(image_path), page)
            name_list.append(name)
            image_path_list.append(str(image_path))
    
    return image_path_list, image_path, name_list
    
def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
                return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

def image_offset(scan_image):
    frmX,toX = 0,200
    frmY,toY = 0,200

    def searchMark(img, pos):

        if pos==0:
            mark = img[frmY:toY, frmX:toX]
            rect = {"x":frmX, "y":frmY}
        elif pos==1:
            mark = img[frmY:toY, img.shape[1]-toX:img.shape[1]-frmX]
            rect = {"x":img.shape[1]-toX, "y":frmY}
        elif pos==2:
            mark = img[img.shape[0]-toY:img.shape[0]-frmY, frmX:toX]
            rect = {"x":frmX, "y":img.shape[0]-toY}
        else:
            mark = img[img.shape[0]-2*toY:img.shape[0]-toY, img.shape[1]-2*toX:img.shape[1]-toX]
            rect = {"x":img.shape[1]-2*toX, "y":img.shape[0]-2*toY}
            
        ret, bin = cv2.threshold(mark, 127, 255, 0)
        contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 2:
            contours = [cont for i,cont in enumerate(contours) if (cont.shape[0] > 30) or (cont.shape[0] < 5) ]
            
        if  len(contours) > 1:
            cnt = contours[len(contours)-1]
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            return 0, 0

        cv2.circle(img,(rect["x"]+cx,rect["y"]+cy), 10, (0,0,255), -1)
        return rect["x"] + cx, rect["y"] + cy

    def searchRegistrationMarks(img):
        cx0,cy0 = searchMark(img,0)
        dx0,dy0 = searchMark(img,1)
        ex0,ey0 = searchMark(img,2)
        fx0,fy0 = searchMark(img,3)

        if (ex0 == 0 & ey0 == 0):
            pts2 = np.float32([[cx0,cy0],[dx0,dy0],[fx0,fy0]])
        elif (dx0 == 0 & dy0 == 0):
            pts2 = np.float32([[fx0,fy0],[ex0,ey0],[cx0,cy0]])
        elif (fx0 == 0 & fy0 == 0):
            pts2 = np.float32([[ex0,ey0],[cx0,cy0],[dx0,dy0]])
        else:
            pts2 = np.float32([[dx0,dy0],[fx0,fy0],[ex0,ey0]])

        return pts2

    src = cv2.imread("C:\\Report_Today\\image_file\\Report_Today_test_01.jpg")
    src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    pts2 = searchRegistrationMarks(src)
    resized_src = cv2.resize(src,(int(src.shape[1]/4), int(src.shape[0]/4)))

    scan = scan_image
    scan = cv2.cvtColor(scan, cv2.COLOR_RGB2GRAY)
    pts1 = searchRegistrationMarks(scan)
    resized_scan = cv2.resize(scan,(int(scan.shape[1]/4), int(scan.shape[0]/4)))

    height,width = src.shape
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(scan,M,(width,height))

    return dst

def file_name_recognize(image):
    name_image = image[310:350, 1150:1350]
    ret, name_image = cv2.threshold(name_image,150,255,cv2.THRESH_BINARY_INV)
    
    model = tf.keras.models.load_model('model_name_recognize.h5',compile=False)
    name_dict = {0:'fujiwara', 1:'fukumoto', 2:'takeda', 3:'asano', 4:'vin', 5:'tanabe', 6:'kobayashi', 7:'saitoh', 8:'yamada', 9:'horikawa'}
    recognize_name = np.argmax(model.predict(np.array(name_image.reshape(1,40,200,1))), axis=1)
    
    name_number = recognize_name[0]
    recognize_name = name_dict.get(name_number)
    return recognize_name

def date_get():
    locale.setlocale(locale.LC_CTYPE, "Japanese_Japan.932")
    now_time = datetime.today()
    return now_time
    
def image_to_summary(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    image = image[496:1090, 1030:1512]
    image = 255-image
    image = cv2.resize(image, (344, 430))
    
    for i in range(8):
        image = np.delete(image, [27*i+i,27*i+i+1,27*i+i+2,27*i+i+3,27*i+i+4,27*i+i+5,27*i+i+6,27*i+i+7,27*i+i+8,27*i+i+9,27*i+i+10,27*i+i+11,27*i+i+12,27*i+i+13,27*i+i+14], 1)
    for j in range(10):
        image = np.delete(image, [27*j+j,27*j+j+1,27*j+j+2,27*j+j+3,27*j+j+4,27*j+j+5,27*j+j+6,27*j+j+7,27*j+j+8,27*j+j+9,27*j+j+10,27*j+j+11,27*j+j+12,27*j+j+13,27*j+j+14], 0)

    images = skimage.util.view_as_blocks(image, (28,28)).copy()
    images = images.reshape((80,784))
    
    zero_image = cv2.imread("C:\\Report_Today\\image_file\\zero_image.jpg")
    zero_image = cv2.cvtColor(zero_image, cv2.COLOR_RGB2GRAY)
    zero_image = zero_image.reshape(1,784)
    
    zero_image_test = zero_image.reshape(28,28)
    
    sum_images = np.sum(images, axis=1)
    images[sum_images<1000] = zero_image
    images = images.astype('float32')/255
    
    images_test = images.reshape((80,28,28))    
    return images

def result_cognization(image,f_name):
    
    model = tf.keras.models.load_model('C:\Report_Today\h5_file\model_fujiwara.h5',compile=False)
    df = DataFrame({'result': list(map(np.argmax, model.predict(image.reshape(80,28,28,1))))})
    df = DataFrame(df.values.reshape(10,8))
    df_max_number = DataFrame({'number': list(map(np.max, model.predict(image.reshape(80,28,28,1))))})
    df_max_number = DataFrame(df_max_number.values.reshape(10,8))
    index_name = []
    for i in range(1,11):
        index_name.append("{0}_{1}".format(f_name,i))

    df.index = index_name
    df_max_number.index = index_name

    max_label = np.argmax(model.predict(image.reshape(80,28,28,1)), axis=1)
    max_number = np.max(model.predict(image.reshape(80,28,28,1)), axis=1)
    max_label = max_label.reshape(80,1)
    max_number = max_number.reshape(80,1)
    max_number = np.where(max_number<0.95, -1, 1)
    image = image.reshape(80,784)
    image_list = np.concatenate([max_label, max_number, image], 1)
    return df, image_list

def result_to_list(df,name):
    data_df = df.values
    data_df = data_df.reshape(10,8)
    arr = ([10,1,10,1,1000,100,10,1])
    data_df = np.multiply(data_df, arr)
    result_list = np.arange(1, 31).reshape(10,3)
    index_name = []
    
    for i in range(10):
        result_list[i,0] = data_df[i,0]+data_df[i,1]
        result_list[i,1] = data_df[i,2]+data_df[i,3]
        result_list[i,2] = data_df[i,4]+data_df[i,5]+data_df[i,6]+data_df[i,7]
        index_name.append("{0}_{1}".format(name,i))
    
    df_result = pd.DataFrame(result_list, index=index_name, columns=['品名','加工','数量'], dtype=int)
    return df_result

def error_determine(image_list_summary,df_result_summary,name_list, file_day):
    df_error_determine = df_result_summary.reset_index(drop=True)
    df_error_determine = df_error_determine.query("品名 != 0 | 加工 != 0 | 数量 != 0")
    df_error = df_error_determine.query("品名 == 0 | 加工 == 0 | 数量 == 0")
    
    df_result_summary = df_result_summary.query("品名 != 0 | 加工 != 0 | 数量 != 0")
    index_number = df_result_summary.index.tolist()

    list_error = df_error.index.tolist()
    image_list_summary = np.array(image_list_summary).reshape(len(name_list)*80,786)
    number = np.arange(len(image_list_summary)).reshape(len(image_list_summary),1)
    image_list_summary = np.hstack((number,image_list_summary))
    
    if len(list_error) != 0:
        for i, error in enumerate(list_error):
            image_list_summary[error*8, 2] = -1
            image_list_summary[error*8+1, 2] = -1
            image_list_summary[error*8+2, 2] = -1
            image_list_summary[error*8+3, 2] = -1
            image_list_summary[error*8+4, 2] = -1
            image_list_summary[error*8+5, 2] = -1
            image_list_summary[error*8+6, 2] = -1
            image_list_summary[error*8+7, 2] = -1
    
    incorrect_list = image_list_summary[np.any(image_list_summary < 0, axis=1)]
    print(incorrect_list.shape)
    
    def selection_incorrect_label(incorrect_list,name_list,image_list_summary):
        root_relearning = tkinter.Toplevel()
    
        root_relearning.title("Selection_Incorrect_Label")
        root_relearning.geometry("700x400")
    
        def relearning_incorrect(incorrect_image):
            
            frame_number = incorrect_list.T
            print(frame_number.shape)
            frame_number = frame_number[0]
            print(frame_number)
            frame_number = frame_number // 8
            frame_number = np.unique(frame_number)
            print(frame_number)
            frame_number = frame_number.tolist()
            for i,number in enumerate(frame_number):
                print(number)
                number_name = number // 10
                incorrect_name = name_list[int(number_name)]
                incorrect_row_number = number % 8
                print(incorrect_name, incorrect_row_number)
                label_text = "Incorrect_Label={0}_{1}".format(incorrect_name, incorrect_row_number)
                frame = tkinter.LabelFrame(main_canvas,text=label_text,width=600,height=100)
                main_canvas.create_window((310,60), window=frame,width=600,height=50)
                frame.pack(side=tkinter.TOP, anchor=tkinter.E)
            
            gen = ImageDataGenerator(
                    rotation_range=30.,
                    width_shift_range=.5,
                    height_shift_range=.5,
                    shear_range=.8,
                    zoom_range=.5)
        
            correct_label = np.zeros((len(incorrect_label),10))
        
            for i in range(10*len(incorrect_label)):
                c = i - (10*int(i/10))
                r = int(i/10)
                if bln[i].get() == True:
                    correct_label[c,r] == 1
                
                incorrect_image = incorrect_image.reshape(len(incorrect_image),28,28,1)

            iters = gen.flow(incorrect_image,correct_label,batch_size=16)
            incorrect_image_batches, correct_label_batches = next(iters)
            incorrect_image = incorrect_image.reshape(len(incorrect_image),28,28,1)
        
            model = models.Sequential()
            model.add(layers.Conv2D(32, (5,5), padding='same', kernel_initializer=initializers.TruncatedNormal(), use_bias=True, activation='relu',input_shape=(28,28,1),name='conv_filter1'))
            model.add(layers.MaxPooling2D((2,2),name='maxpooling'))
            model.add(layers.Flatten(name='flatten'))
            model.add(layers.Dense(512, activation='relu',kernel_initializer=initializers.TruncatedNormal(),name='hidden'))
            model.add(layers.Dropout(rate=0.5, name='dropout'))
            model.add(layers.Dense(10,activation='softmax',name='softmax'))
            model.summary()
        
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        
            history = model.fit(incorrect_image, correct_label, batch_size=1, epochs=10)
        
            model.save('model_fujiwara.h5')
        
            relearning_image_path = Path("C:\\Report_Today\\image_file"+"\\"+relearning_files_entry.get())
            relearning_images = image_to_summary(relearning_image_path)
            df_today = result_cognization(relearning_images)
            result_df, result_list = result_to_list(df_today)
            tree.heading("#0", text = "メンバー")
            tree.heading("product", text = "品名")
            tree.heading("work", text = "加工")
            tree.heading("number", text = "数量")
            for i in range(len(result_list)):
                tree.insert("", "end", text="work", values=result_list[i])
        
            root_relearning.destroy()
        
        def destroy_window(incorrect_list,df_result_summary):
            for i, incorrect in enumerate(incorrect_list):
                image_list_summary[int(incorrect[0]), 1] = combobox_plot.get()
                image_list_summary[int(incorrect[0]), 2] = 1
                
            name_number = []
            for name in enumerate(name_list):
                for i in range(80):
                    j = i // 8
                    k = i % 10
                    name_number.append("{0}_{1}".format(name,j))
                    name_number.append(k)
            else:
                name_number = np.array(name_number).reshape(len(name_list)*80,2)
        
            report_list = ['品名','加工','数量']
    
            product_fujiwara = pd.Series([np.nan,'エクシズ','エクシズ_金型1','エクシズ_金型2','エクシズ_金型3'])
            product_fukumoto = pd.Series([np.nan,'Pitchford FL','Pitchford ST','ZX IN FL','ZX IN ST','ZX OUT FL','ZX OUT ST','ZX FIBER FL','ZX FIBER ST','Blockman FL','Blockman CHN','S Defensive FL','S Defensive ST','a','b','c','d','e','f','g','h','i','j','k','l'])
            product_takeda = pd.Series([np.nan,'Pitchford FL','Pitchford ST','ZX IN FL','ZX IN ST','ZX OUT FL','ZX OUT ST','ZX FIBER FL','ZX FIBER ST','Blockman FL','Blockman CHN','S Defensive FL','S Defensive ST'])
            product_asano = pd.Series([np.nan,'Pitchford FL','Pitchford ST','ZX IN FL','ZX IN ST','ZX OUT FL','ZX OUT ST','ZX FIBER FL','ZX FIBER ST','Blockman FL','Blockman CHN','S Defensive FL','S Defensive ST'])
            product_vin =pd.Series([np.nan,])
            product_tanabe = pd.Series([np.nan,])
            product_kobayashi = pd.Series([np.nan,])
            product_saitoh = pd.Series([np.nan,])
            product_yamada = pd.Series([np.nan,])
            product_horikawa = pd.Series([np.nan,])
    
            work_fujiwara = pd.Series([np.nan,'練り','プレス','打球テスト'])
            work_fukumoto = pd.Series([np.nan,'グリップ接着','裏削り','グリップ周り削り','不良補修','ブレード周り削り'])
            work_takeda = pd.Series([np.nan,'グリップ裏削り','寸切り','鉛貼り','厚さ調整','合板','グリップ回り削り'])
            work_asano = pd.Series([np.nan,'グリップ接着','型抜き','型合わせ'])
            work_vin = pd.Series([np.nan,])
            work_tanabe = pd.Series([np.nan,])
            work_kobayashi = pd.Series([np.nan,])
            work_saitoh = pd.Series([np.nan,])
            work_yamada = pd.Series([np.nan,])
            work_horikawa = pd.Series([np.nan,])
    
            work_list_summary = pd.concat([work_fujiwara,work_fukumoto,work_takeda,work_asano,work_vin,work_tanabe,work_kobayashi,work_saitoh,work_yamada,work_horikawa],axis=1)
            work_list = work_list_summary.values.tolist()
    
            product_list_summary = pd.concat([product_fujiwara,product_fukumoto,product_takeda,product_asano,product_vin,product_tanabe,product_kobayashi,product_saitoh,product_yamada,product_horikawa],axis=1)
            product_list = product_list_summary.values.tolist()
            list_df = df_result_summary.values.tolist()
            
            for i, number in enumerate(index_number):
                name_dict = {'fujiwara':0, 'fukumoto':1, 'takeda':2, 'asano':3, 'vin':4, 'tanabe':5, 'kobayashi':6, 'saitoh':7, 'yamada':8, 'horikawa':9}
                name_number = name_dict.get('{0}'.format(number[:-2]))
                
                list_df[i][0] = product_list[list_df[i][0]][name_number]
                list_df[i][1] = work_list[list_df[i][1]][name_number]
        
            index = pd.MultiIndex.from_product([[file_day],index_number])
            df_result_summary = pd.DataFrame(list_df,index=index,columns=report_list,dtype=str)
            df_result_summary.index.names = ['日付','名前']
            df_result_summary = df_result_summary.reset_index()
            df_result_summary.to_csv('C:\\Report_Today\\csv_file\\result_today_test.csv', sep=',', encoding='cp932')
            result_list = df_result_summary.values.tolist()
            
            tree.heading("date", text = "日付")
            tree.heading("member_number", text = "名前")
            tree.heading("product", text = "品名")
            tree.heading("work", text = "加工")
            tree.heading("number", text = "数量")
            for i in tree.get_children():
                tree.delete(i)
            for i in range(len(result_list)):
                tree.insert("", "end", text="work", values=result_list[i])
            
            root_relearning.destroy()
        
        main_canvas = tkinter.Canvas(root_relearning, width=200, height=400)
        main_canvas.pack(fill=tkinter.BOTH, expand=1, padx=10, pady=10, side=tkinter.TOP)
        main_bar = ttk.Scrollbar(main_canvas, orient=tkinter.VERTICAL,command=main_canvas.yview)
        main_bar.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        main_bar.configure(command=main_canvas.yview)
        main_canvas.configure(yscrollcommand=main_bar.set)
        main_canvas.configure(scrollregion=(0,0,100,2000))
        frame = tkinter.Frame(main_canvas)
        main_canvas.create_window((0,0), window=frame, anchor="nw")
        main_button = tkinter.Button(root_relearning, text="訂正完了",command=lambda: destroy_window(incorrect_list,df_result_summary))
        main_button.pack(side=tkinter.BOTTOM, anchor=tkinter.E)
        frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
        
        incorrect_list = incorrect_list.tolist()
        for i, incorrect in enumerate(incorrect_list):
            label, image = np.hsplit(np.array(incorrect), [3])
            
            incorrect_name = label[0] // 80
            incorrect_name = name_list[int(incorrect_name)]
            incorrect_row_number = (label[0] % 80) // 8
            incorrect_column_number = (label[0] % 80) % 8
            
            fig = plt.Figure(figsize=(.5,.5))
            canvas = FigureCanvasTkAgg(fig, frame)
        
            combobox_plot = ttk.Combobox(frame, values=list(range(10)), width=8)
            
            c = i % 8
            r = i // 8
            
            subplot = fig.add_subplot(1,1,1)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.set_title("{0}".format(label[1]),size=5)
            subplot.imshow(image.reshape((28,28)), vmin=0, vmax=1, cmap=plt.cm.gray_r)
        
            fig.canvas.draw()
            canvas.get_tk_widget().grid(row=2*r,column=c,padx=2, pady=1)
            combobox_plot.grid(row=2*r+1,column=c, padx=1)
            combobox_plot.set("数値選択")
    
        root.mainloop()
        
    
    if len(incorrect_list) != 0:
        res = tkinter.messagebox.showerror("Error", "エラーがあります")
        if res == 'ok':
            selection_incorrect_label(incorrect_list, name_list,image_list_summary)
    
    name_number = []
    for name in enumerate(name_list):
        for i in range(80):
            j = i // 8
            k = i % 10
            name_number.append("{0}_{1}".format(name,j))
            name_number.append(k)
    else:
        name_number = np.array(name_number).reshape(len(name_list)*80,2)
        
    report_list = ['品名','加工','数量']
    
    product_fujiwara = pd.Series([np.nan,'エクシズ','エクシズ_金型1','エクシズ_金型2','エクシズ_金型3'])
    product_fukumoto = pd.Series([np.nan,'Pitchford FL','Pitchford ST','ZX IN FL','ZX IN ST','ZX OUT FL','ZX OUT ST','ZX FIBER FL','ZX FIBER ST','Blockman FL','Blockman CHN','S Defensive FL','S Defensive ST'])
    product_takeda = pd.Series([np.nan,'Pitchford FL','Pitchford ST','ZX IN FL','ZX IN ST','ZX OUT FL','ZX OUT ST','ZX FIBER FL','ZX FIBER ST','Blockman FL','Blockman CHN','S Defensive FL','S Defensive ST'])
    product_asano = pd.Series([np.nan,'Pitchford FL','Pitchford ST','ZX IN FL','ZX IN ST','ZX OUT FL','ZX OUT ST','ZX FIBER FL','ZX FIBER ST','Blockman FL','Blockman CHN','S Defensive FL','S Defensive ST'])
    product_vin =pd.Series([np.nan,])
    product_tanabe = pd.Series([np.nan,])
    product_kobayashi = pd.Series([np.nan,])
    product_saitoh = pd.Series([np.nan,])
    product_yamada = pd.Series([np.nan,])
    product_horikawa = pd.Series([np.nan,])
    
    work_fujiwara = pd.Series([np.nan,'練り','プレス','打球テスト'])
    work_fukumoto = pd.Series([np.nan,'グリップ接着','裏削り','グリップ周り削り','不良補修','ブレード周り削り'])
    work_takeda = pd.Series([np.nan,'グリップ裏削り','寸切り','鉛貼り','厚さ調整','合板','グリップ回り削り'])
    work_asano = pd.Series([np.nan,'グリップ接着','型抜き','型合わせ'])
    work_vin = pd.Series([np.nan,])
    work_tanabe = pd.Series([np.nan,])
    work_kobayashi = pd.Series([np.nan,])
    work_saitoh = pd.Series([np.nan,])
    work_yamada = pd.Series([np.nan,])
    work_horikawa = pd.Series([np.nan,])
    
    work_list_summary = pd.concat([work_fujiwara,work_fukumoto,work_takeda,work_asano,work_vin,work_tanabe,work_kobayashi,work_saitoh,work_yamada,work_horikawa],axis=1)
    work_list = work_list_summary.values.tolist()
    
    product_list_summary = pd.concat([product_fujiwara,product_fukumoto,product_takeda,product_asano,product_vin,product_tanabe,product_kobayashi,product_saitoh,product_yamada,product_horikawa],axis=1)
    product_list = product_list_summary.values.tolist()
    
    list_df = df_result_summary.values.tolist()
    
    for i, number in enumerate(index_number):
        name_dict = {'fujiwara':0, 'fukumoto':1, 'takeda':2, 'asano':3, 'vin':4, 'tanabe':5, 'kobayashi':6, 'saitoh':7, 'yamada':8, 'horikawa':9}
        name_number = name_dict.get('{0}'.format(number[:-2]))
        list_df[i][0] = product_list[list_df[i][0]][name_number]
        list_df[i][1] = work_list[list_df[i][1]][name_number]
        
    index = pd.MultiIndex.from_product([[file_day],index_number])
    df_result_summary = pd.DataFrame(list_df,index=index,columns=report_list,dtype=str)
    
    df_result_summary.index.names = ['日付','名前']
    df_result_summary = df_result_summary.reset_index()
    
    return df_result_summary
    

def select_date():
    root_date = tkinter.Toplevel()
    
    root_date.title("Select_Date")
    root_date.geometry("300x300")
    
    now_time = date_get()
    
    frame_month = ttk.Frame(root_date)
    frame_month.pack(pady=5)
    
    now_day = [now_time.strftime('%Y%m%d')]
    
    def switch_previous_month(now_day, frame_day):
        now_time = datetime.strptime(now_day[0], '%Y%m%d')
        now_time -= relativedelta(months=1)
        text_current_year.set(str(now_time.year))
        text_current_month.set(str(now_time.month))
        now_day.insert(0, now_time.strftime('%Y%m%d'))
        children = frame_day.winfo_children()
        for child in children:
            child.destroy()
        calendar_list(now_time)
        return now_day
    
    previous_month = ttk.Button(frame_month, text="<", command=lambda: switch_previous_month(now_day, frame_day), width=3)
    previous_month.pack(side = "left", padx=10)
    
    def switch_next_month(now_day, frame_day):
        now_time = datetime.strptime(now_day[0], '%Y%m%d')
        now_time += relativedelta(months=1)
        text_current_year.set(str(now_time.year))
        text_current_month.set(str(now_time.month))
        now_day.insert(0, now_time.strftime('%Y%m%d'))
        children = frame_day.winfo_children()
        for child in children:
            child.destroy()
        calendar_list(now_time)
        return now_day
    
    next_month = ttk.Button(frame_month, text=">", command=lambda: switch_next_month(now_day, frame_day), width=3)
    next_month.pack(side="right", padx=10)
    
    text_current_year = tkinter.StringVar()
    text_current_year.set(str(now_time.year))
    current_year = ttk.Label(frame_month, textvariable=text_current_year)
    current_year.pack(side="left")
    
    text_current_month = tkinter.StringVar()
    text_current_month.set(str(now_time.month))
    current_month = ttk.Label(frame_month, textvariable=text_current_month)
    current_month.pack(side="left")
    
    frame_week = ttk.Frame(root_date)
    frame_week.pack(pady=1)
    label_mon = ttk.Label(frame_week, text="Mon")
    label_mon.grid(column=0, row=0)
    label_tue = ttk.Label(frame_week, text="Tue")
    label_tue.grid(column=1, row=0)
    label_wed = ttk.Label(frame_week, text="Wed")
    label_wed.grid(column=2, row=0)
    label_thu = ttk.Label(frame_week, text="Thu")
    label_thu.grid(column=3, row=0)
    label_fri = ttk.Label(frame_week, text="Fri")
    label_fri.grid(column=4, row=0)
    label_sta = ttk.Label(frame_week, text="Sta")
    label_sta.grid(column=5, row=0)
    label_sun = ttk.Label(frame_week, text="Sun")
    label_sun.grid(column=6, row=0)
    
    frame_day =ttk.Frame(root_date, height=200)
    frame_day.pack(pady=1)
    
    
    def calendar_list(now_time):
        cal = calendar.Calendar()
        days = cal.monthdayscalendar(now_time.year, now_time.month)
        
        def switch_date(dat):
            def x():
                entry_date.set("{:04d}/{:02d}/{}".format(int(text_current_year.get()),int(text_current_month.get()),str(dat)))
            return x
        
        day = {}
        for i in range(0,42):
            c = i - (7*int(i/7))
            r = int(i/7)
            try:
                if days[r][c] != 0:
                    dat = str("{:02d}".format(int(days[r][c])))
                    day[i] = ttk.Button(frame_day,text=days[r][c],command=switch_date(dat), width=3)
                    day[i].grid(column=c,row=r)
            except:
                break
        return days
    
    calendar_list(now_time)
    
    def decide_date():
        select_date_button["text"] = entry_date.get()
        root_date.destroy()
        
    frame_decide_date = ttk.Frame(root_date)
    frame_decide_date.pack(pady=1)
    entry_date = tkinter.StringVar()
    select_date = tkinter.Entry(frame_decide_date, textvariable=entry_date, width=30)
    decide_date_button = ttk.Button(frame_decide_date, text="日付選択", command=decide_date)
    select_date.grid(column=6,row=0)
    decide_date_button.grid(column=7,row=0)
    
    root.mainloop()

def progressbar():
    root_progressbar = tkinter.Toplevel()
    root_progressbar.title("Progress")
    root_progressbar.geometry("300x100")
    
    progressbar = ttk.Progressbar(root_progressbar,length=200,mode='indeterminate')
    progressbar.configure(maximum=20, value=0, mode='indeterminate')
    progressbar.grid(row=0, column=0, sticky=(tkinter.N,tkinter.E,tkinter.S,tkinter.W))
    progressbar.start(interval=10)
    
    progress_value = 0
    progress_value = progress_value + 1
    progressbar.configure(value=progress_value)
    
    root.mainloop()

def list_to_xlsx():
    df = pd.read_csv('C:\\Report_Today\\csv_file\\result_today_test.csv',encoding='cp932')
    report_today_list = df.to_excel("C:\\Report_Today\\xlsx_file\\report_today_test.xlsx", encoding='cp932')
    report_today_summary_df = pd.read_excel('C:\\Report_Today\\xlsx_file\\report_today_summary.xlsx')
    report_today_summary_df = pd.concat([report_today_summary_df, df], axis=0)
    report_today_summary_df = report_today_summary_df.drop(columns=report_today_summary_df.columns[[0]])
    report_today_summary_df.to_excel('C:\\Report_Today\\xlsx_file\\report_today_summary.xlsx', encoding='cp932')
    res = tkinter.messagebox.showinfo("完了","Excel出力完了です!")
    if res == 'ok':
        sys.exit()
    return report_today_list

def button_press():
    search_member = dictionary_combobox_member.get()
    search_product = dictionary_combobox_product.get()
    search_work = dictionary_combobox_work.get()
    if search_member != "メンバー選択":
        report_today_summary_df = report_today_summary_df.query("名前" == search_member)

def search(report_today_summary_df):
    for i in tree.get_children():
        tree.delete(i)
    search_member = dictionary_combobox_member.get()
    search_product = dictionary_combobox_product.get()
    search_work = dictionary_combobox_work.get()
    if search_member != "メンバー選択":
        report_today_summary_df = report_today_summary_df[report_today_summary_df['名前'] == search_member]
    if search_product != "品名選択":
        report_today_summary_df = report_today_summary_df[report_today_summary_df['品名'] == search_product]
    if search_work != "加工名選択":
        report_today_summary_df = report_today_summary_df[report_today_summary_df['加工'] == search_work]
    report_today_summary_df = report_today_summary_df.sort_values('加工')
    report_today_summary_df = report_today_summary_df.sort_values('品名')
    report_today_summary_df = report_today_summary_df.sort_values('名前')
    report_today_summary_df = report_today_summary_df.sort_values('日付')
    report_today_summary_df = report_today_summary_df.drop(report_today_summary_df.columns[[0]],axis=1)
    report_today_summary_list = report_today_summary_df.values.tolist()
    
    tree.heading("date", text = "日付")
    tree.heading("member_number", text = "名前")
    tree.heading("product", text = "品名")
    tree.heading("work", text = "加工")
    tree.heading("number", text = "数量")
    
    for i in range(len(report_today_summary_list)):
        tree.insert("", "end", text="work", values=report_today_summary_list[i])

if __name__ == "__main__":

    root = tkinter.Tk()

    root.title("Report_Today")
    root.geometry("700x400")

    frame_report = tkinter.LabelFrame(root,text='日報更新')
    folder_label = ttk.Label(frame_report, text="フォルダ指定 ->")
    entry2 = tkinter.StringVar()
    folder_entry = ttk.Entry(frame_report, textvariable=entry2, width=50)
    folder_button = ttk.Button(frame_report, text="参照", command=filedialog_clicked)
    run_button = ttk.Button(frame_report, text="実行", command=pdf_to_list)
    frame_search = tkinter.LabelFrame(root,text='履歴検索')
    report_today_summary_df = pd.read_excel('C:\\Report_Today\\xlsx_file\\report_today_summary.xlsx')
    report_today_summary_value = report_today_summary_df.values.tolist()
    for i in range(len(report_today_summary_value)):
        report_today_summary_value[i][2] = str(report_today_summary_value[i][2])[:-2]
    report_today_summary_df = pd.DataFrame(report_today_summary_value,columns=report_today_summary_df.columns,dtype=str)
    report_today_groupby_member = report_today_summary_df.groupby("名前").min()
    report_today_groupby_product = report_today_summary_df.groupby("品名").min()
    report_today_groupby_work = report_today_summary_df.groupby("加工").min()
    member_list = list(report_today_groupby_member.index)
    member_list.insert(0,"メンバー選択")
    dictionary_combobox_member = ttk.Combobox(frame_search, values=member_list, width=20)
    dictionary_combobox_member.set("メンバー選択")
    product_list = list(report_today_groupby_product.index)
    product_list.insert(0,"品名選択")
    dictionary_combobox_product = ttk.Combobox(frame_search, values=product_list, width=20)
    dictionary_combobox_product.set("品名選択")
    work_list = list(report_today_groupby_work.index)
    work_list.insert(0,"加工名選択")
    dictionary_combobox_work = ttk.Combobox(frame_search, values=work_list, width=20)
    dictionary_combobox_work.set("加工名選択")
    search_button = ttk.Button(frame_search, text="検索", command=lambda: search(report_today_summary_df))
    tree = ttk.Treeview(root, columns=('date','member_number','product', 'work', 'number'))
    scroll_y = ttk.Scrollbar(tree, orient=tkinter.VERTICAL, command=tree.yview)
    tree.column('#0', width=0, minwidth=0)
    tree.column("date", width=60, minwidth=60)
    tree.column("member_number", width=100, minwidth=50)
    tree.column("product", width=180, minwidth=100)
    tree.column("work", width=150, minwidth=100)
    tree.column("number", width=100, minwidth=100)
    output_button = ttk.Button(root, text="Excel出力", command=list_to_xlsx)
    select_date_button = ttk.Button(frame_report, text="日付選択", command=select_date)

    frame_report.pack(padx=10,pady=10,fill=tkinter.X,anchor=tkinter.N)
    folder_label.grid(row=0,column=0,padx=6,pady=6)
    folder_entry.grid(row=0,column=1,padx=6,pady=6)
    folder_button.grid(row=0,column=2,padx=6,pady=6)
    select_date_button.grid(row=0,column=3,padx=6,pady=6)
    run_button.grid(row=0,column=4,padx=6,pady=6)
    frame_search.pack(padx=10,pady=5,fill=tkinter.X)
    dictionary_combobox_member.grid(row=0,column=0,padx=6,pady=6)
    dictionary_combobox_product.grid(row=0,column=1,padx=40,pady=6)
    dictionary_combobox_work.grid(row=0,column=2,padx=20,pady=6)
    search_button.grid(row=0,column=3,padx=30,pady=6)
    tree.pack(padx=10,pady=10,fill=tkinter.BOTH,ipady=70)
    scroll_y.pack(anchor=tkinter.E,fill=tkinter.Y,side=tkinter.RIGHT)
    tree.configure(yscrollcommand=scroll_y.set)
    output_button.pack(padx=10,pady=5,anchor=tkinter.S,side=tkinter.RIGHT)

    root.mainloop()
    