#スキャンしたpdfファイルを数値認識し、Excelで保存するプログラム
import os, sys
from pathlib import Path
from pdf2image import convert_from_path
import numpy as np
import cv2
from datetime import datetime
import locale
from PIL import Image, ImageDraw
import pyocr
import pyocr.builders
import xlwings as xw

#複数のpdfファイルを分割し、JPEGファイルに変換する　⇒　参考サイト：https://gammasoft.jp/blog/convert-pdf-to-image-by-python/
def pdf_to_image():
    #pdfからJPEGに変換するライブラリが保存されているディレクトリ
    poppler_dir = Path("C:\\Shimizu_Data\\poppler").parent.absolute() / "poppler\\bin"
    os.environ["PATH"] += os.pathsep + str(poppler_dir)

    #入力するpdfファイルの保存パス
    pdf_path = Path("C:\\Shimizu_data\\pdf_file\\scan.pdf")
    #出力するJPEGファイルの保存先ディレクトリ
    image_dir = Path("C:\\Shimizu_Data\\image_file")
    
    #pdfからJPEGに変換するコード、コード内の"200"は変換時の画像の粗さ
    image = convert_from_path(str(pdf_path), 200, fmt="JPEG")
    #保存するJPEGファイルのパスをリストボックスを作成する、
    image_path_list = []
    #プログラムを起動した日付を取得する
    now_time = date_get()
    
    #JPEGに変換したファイルを一枚ずつ位置補正とファイルの名づけをする
    for i, page in enumerate(image):
        #ファイルの系統を変更する
        page = np.asarray(page)
        #ファイルの位置補正、下記の定義した関数を使う
        page = image_offset(page)
        #保存するファイル名を作成する、"日付_Experiment_Condition_番号(スキャンの順番に1,2,3,,,と続く).jpg"
        file_name = now_time.strftime('%Y%m%d') + "_Experiment_Condition" + "_{0}".format(i+1) + ".jpg"
        image_path = image_dir / file_name
        cv2.imwrite(str(image_path), page)
        #作成したJPEGファイルのパスをimage_path_listに追加する
        image_path_list.append(str(image_path))
    
    return image_path_list

#スキャンの位置補正する　⇒　参考サイト:https://qiita.com/danjiro/items/985afccd07722d4c21e9
def image_offset(scan_image):
    #特徴点を検索する範囲を決める
    frmX,toX = 0,200
    frmY,toY = 0,200

    #指定した範囲から特徴点を探す
    def searchMark(img, pos):
        
        #左上の特徴点を検索する
        if pos==0:
            mark = img[frmY:toY, frmX:toX]
            rect = {"x":frmX, "y":frmY}
        #右上の特徴点を検索する
        elif pos==1:
            mark = img[frmY:toY, img.shape[1]-toX:img.shape[1]-frmX]
            rect = {"x":img.shape[1]-toX, "y":frmY}
        #左下の特徴点を検索する
        elif pos==2:
            mark = img[img.shape[0]-toY:img.shape[0], frmX:toX]
            rect = {"x":frmX, "y":img.shape[0]-toY}
        #右下の特徴点を検索する
        else:
            mark = img[img.shape[0]-2*toY:img.shape[0]-toY, img.shape[1]-2*toX:img.shape[1]-toX]
            rect = {"x":img.shape[1]-2*toX, "y":img.shape[0]-2*toY}
            
        #2値化し、検索範囲にある輪郭をプロットする
        ret, bin = cv2.threshold(mark, 127, 255, 0)
        contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)

        #検索範囲に特徴点があったら、その重心の位置をcxとcyとする
        if  len(contours) > 1:
            cnt = contours[len(contours)-1]
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        #検索範囲に特徴点がなければ、0とする
        else:
            return 0, 0

        #特徴点に点をプロットする
        cv2.circle(img,(rect["x"]+cx,rect["y"]+cy), 10, (0,0,255), -1)
        return rect["x"] + cx, rect["y"] + cy
    
    #4箇所から検索した特徴点の重心を集計する
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

    #基準データを読み取る
    src = cv2.imread("C:\\Shimizu_Data\\learning_image\\shimizu_sample.JPG")
    src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    #基準データの特徴点の位置を検索する
    pts2 = searchRegistrationMarks(src)

    #スキャンしたデータを読み取る
    scan = scan_image
    scan = cv2.cvtColor(scan, cv2.COLOR_RGB2GRAY)
    #スキャンしたデータの特徴点の位置を検索する
    pts1 = searchRegistrationMarks(scan)

    #スキャンしたデータの特徴点が基準データと一致するように画像を変換する
    height,width = src.shape
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(scan,M,(width,height))

    return dst

#プログラムを起動する日付の取得
def date_get():
    locale.setlocale(locale.LC_CTYPE, "Japanese_Japan.932")
    now_time = datetime.today()
    return now_time

#エクセルに認識データを入力する　⇒　参考サイト：https://imagisi.com/auto-text-to-excel/
def image_to_excel(image, wb, i):
    #OCRで画像を文字列に変換する
    builder = pyocr.builders.DigitLineBoxBuilder(tesseract_layout=6)
    result = tool.image_to_string(image, lang="eng", builder=builder)
    
    #認識を確認するため、画像を表示する
    for re in result:
        #画像に認識数値を囲う　⇒　参考サイト：https://note.nkmk.me/python-pillow-imagedraw/
        draw = ImageDraw.Draw(image)
        draw.rectangle((re.position[0],re.position[1]), outline=0)
    else:
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

def comp_image(image):
    #画像を白黒に変更する
    image = image.convert("L")
    #画像を２値化する　⇒　参考サイト：https://qiita.com/pashango2/items/145d858eff3c505c100a
    image.point(lambda x: 0 if x < 254 else 255)
    #認識する画像をトリミングする
    img_1 = image.crop((472,353,1007,1031))
    #英字部分以外の範囲をトリミングする
    img_1_1 = img_1.crop((0,0,490,370))
    img_1_2 = img_1.crop((0,414,490,504))
    img_1_3 = img_1.crop((0,557,490,674))
    #余計な範囲を切り取った領域を連結する　⇒　参考サイト：https://note.nkmk.me/python-pillow-concat-images/
    dst = Image.new('L', (img_1.width, img_1_1.height+img_1_2.height+img_1_3.height))
    dst.paste(img_1_1, (0,0))
    dst.paste(img_1_2, (0,img_1_1.height))
    dst.paste(img_1_3, (0,img_1_1.height+img_1_2.height))
    
    #エクセルデータを新規作成
    wb = xw.Book("C:\\Shimizu_Data\\xlsx_file\\test.xlsx")
    
    for i in range(7):
        #１行分の範囲をトリミングする
        im = dst.crop((70*i,0,70*(i+1)+3,dst.height))
        #１行分の数値を認識し、エクセルに変換する
        image_to_excel(im,wb,i)
        
    now_time = date_get()
    excel_dir = Path("C:\\Shimizu_Data\\xlsx_file")
    excel_name = now_time.strftime('%Y%m%d') + "_Experiment_Condition" + "_{0}".format(i+1) + ".xlsx"
    image_path = excel_dir / excel_name
    #作成したエクセルデータを保存する
    wb.save(str(image_path))
    #システムを終了する
    sys.exit(0)
    
#インストール済みのTesseractのパスを通す　⇒　参考サイト：https://gammasoft.jp/blog/ocr-by-python/
path_tesseract = "C:\\Users\\fujihara\\AppData\\Local\\Tesseract-OCR"
if path_tesseract not in os.environ["PATH"].split(os.pathsep):
    os.environ["PATH"] += os.pathsep + path_tesseract

#OCRエンジンを取得する
tools = pyocr.get_available_tools()
tool = tools[0]

if len(tools) == 0:
    print("OCRツールが見つかりません")
    sys.exit(1)

#pdfファイルをjpgに変換する
image_path_list = pdf_to_image()
for i, path in enumerate(image_path_list):
    #画像をエクセルに変換する
    comp_image(Image.open(path))