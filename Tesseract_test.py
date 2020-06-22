import os
import sys
from pathlib import Path
from PIL import Image,ImageDraw
import pyocr
import pyocr.builders
import xlwings as xw
import tkinter
from tkinter import ttk
from tkinter import filedialog
from datetime import datetime
from dateutil.relativedelta import relativedelta
import locale
import calendar

def image_to_excel(image, wb, i):
    #インストール済みのTesseractのパスを通す
    path_tesseract = "C:\\Users\\fujihara\\AppData\\Local\\Tesseract-OCR"
    if path_tesseract not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + path_tesseract

    #OCRエンジンを取得する
    tools = pyocr.get_available_tools()
    tool = tools[0]

    if len(tools) == 0:
        print("OCRツールが見つかりません")
        sys.exit(1)
    #OCRで画像を文字列に変換する
    builder = pyocr.builders.DigitLineBoxBuilder(tesseract_layout=6)
    result = tool.image_to_string(image, lang="eng", builder=builder)
    
    #認識数値の部分を囲う
    for re in result:
        print("{}{}".format(re.content, re.position))
        draw = ImageDraw.Draw(image)
        draw.rectangle((re.position[0],re.position[1]), outline=0)
    else:
        #確認用に画像を表示する
        image.show()
        
    #保存した文字列をエクセルに入力する
    y = 2
    for data in result:
        data = "{}".format(data.content)
        if data == "#":
            continue
        split_data = data.split()
        if split_data == []:
            continue
        wb.sheets[0].range((y,i+3)).value = split_data
        y += 1

def comp_image():
    #選択したファイルパスを取得する
    image_path = Path(entry2.get())
    image = Image.open(image_path)
    #画像を白黒に変更する
    image = image.convert("L")
    #画像を２値化する
    image.point(lambda x: 0 if x < 254 else 255)
    #認識する画像をトリミングする
    img_1 = image.crop((472,353,1007,1031))
    #英字部分以外の範囲をトリミングする
    img_1_1 = img_1.crop((0,0,490,370))
    img_1_2 = img_1.crop((0,414,490,504))
    img_1_3 = img_1.crop((0,557,490,674))
    #余計な範囲を切り取った領域を連結する
    dst = Image.new('L', (img_1.width, img_1_1.height+img_1_2.height+img_1_3.height))
    dst.paste(img_1_1, (0,0))
    dst.paste(img_1_2, (0,img_1_1.height))
    dst.paste(img_1_3, (0,img_1_1.height+img_1_2.height))

    #エクセルデータを新規作成
    wb = xw.Book("C:\\Shimizu_Data\\xlsx_file\\test.xlsx")
    
    for i in range(7):
        #１行分の範囲を切り取る
        im = dst.crop((70*i,0,70*(i+1)+3,dst.height))
        
        #１行分の数値を認識し、エクセルに変換する
        image_to_excel(im,wb,i)
        
    #選択した日付を保存先のファイル名に入れる
    excel_dir = Path("C:\\Shimizu_Data\\xlsx_file")
    if select_date_button["text"] == "日付選択":
        now_time = date_get()
        file_name = now_time.strftime('%Y%m%d') + "_report" + ".xlsx"
        excel_path = excel_dir / file_name
    else:
        select_day = select_date_button["text"].replace("/","")
        file_name = select_day + "_report" + ".xlsx"
        excel_path = excel_dir / file_name
    
    #作成したエクセルデータを保存する
    wb.save(str(excel_path))
    #エクセルアプリケーションを削除する
    wb.app.quit()
    #システムを正常終了する
    sys.exit(0)

def filedialog_clicked():
    fTyp = [("", "*")]
    iFile = os.path.abspath(os.path.dirname(__file__))
    iFilePath = filedialog.askopenfilename(filetype = fTyp, initialdir = iFile)
    entry2.set(iFilePath)

def date_get():
    locale.setlocale(locale.LC_CTYPE, "Japanese_Japan.932")
    now_time = datetime.today()
    return now_time

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
    
    
if __name__ == "__main__":

    root = tkinter.Tk()

    root.title("Test")
    root.geometry("700x100")

    frame_report = tkinter.LabelFrame(root,text='エクセル入力')
    folder_label = ttk.Label(frame_report, text="フォルダ指定 ->")
    entry2 = tkinter.StringVar()
    folder_entry = ttk.Entry(frame_report, textvariable=entry2, width=50)
    folder_button = ttk.Button(frame_report, text="参照", command=filedialog_clicked)
    run_button = ttk.Button(frame_report, text="実行", command=comp_image)
    select_date_button = ttk.Button(frame_report, text="日付選択", command=select_date)

    frame_report.pack(padx=10,pady=10,fill=tkinter.X,anchor=tkinter.N)
    folder_label.grid(row=0,column=0,padx=6,pady=6)
    folder_entry.grid(row=0,column=1,padx=6,pady=6)
    folder_button.grid(row=0,column=2,padx=6,pady=6)
    select_date_button.grid(row=0,column=3,padx=6,pady=6)
    run_button.grid(row=0,column=4,padx=6,pady=6)

    root.mainloop()