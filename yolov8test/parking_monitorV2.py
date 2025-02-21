import cv2
import pytesseract
import torch
import csv
import smtplib
import time
import datetime
import os
import tkinter as tk
from tkinter import messagebox
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import re
import matplotlib.pyplot as plt
import numpy as np

# load .env file
load_dotenv('private_info.env')

# 攝影機IP, RTSP 帳戶
ipcam_adress = os.getenv("CAM1_IP")
rtsp_account = os.getenv("RTSP_ID")
rtsp_password = os.getenv("RTSP_PWD")

# 設定車牌辨識工具路徑（適用於 Windows）
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# # 載入 YOLOv5 模型
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model
model = YOLO("runs/detect/train13/weights/best.pt")  # load a custom model

# 設定監控區域攝影機
cap = cv2.VideoCapture(f"rtsp://{rtsp_account}:{rtsp_password}@{ipcam_adress}:554/stream2")  # 0 代表第一個攝影機

detected_plates = set()

def send_email(plate_number, timestamp):
    sender_email = os.getenv("EMAIL_USER")
    receiver_email = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")
    
    subject = "車牌偵測通知"
    body = f"偵測到車牌 {plate_number} 於 {timestamp} 進入停車區"
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"通知已發送至 {receiver_email}")
    except Exception as e:
        print("Email 發送失敗: ", e)


def recognize_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('thresh_img', thresh)
    text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    print(text)
    return text.strip()


def log_plate(plate_number):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if plate_number and plate_number not in detected_plates:
        detected_plates.add(plate_number)
        with open("parking_log.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([plate_number, timestamp])
        print(f"記錄: {plate_number} - {timestamp}")
        send_email(plate_number, timestamp)


def detect_parking():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # print('frame.shape:', frame.shape) = (360, 640, 3)
        cv2.imshow("Parking Monitor", frame)
        import time
        nowTime = int(time.time()) # 取得現在時間
        if nowTime%60==0:
            # # 使用 YOLO 偵測車牌
            results = model(frame, show=True)
            for detection in results[0].boxes:
                x1, y1, x2, y2 = detection.xyxy[0].tolist()
                print(x1, x2, y1, y2)
                conf = detection.conf[0].tolist()
                if conf > 0.5:  # 只處理高信心度的偵測結果
                    plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    plates = recognize_plate(plate_crop)
                    # print(plates)
                    
                    # 驗證車牌格式
                    pattern = r"^[A-Z0-9]{6,8}$"
                    if re.match(pattern, plates):
                        print("有效車牌：", plates)
                        log_plate(plates)
        
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def start_gui():
    root = tk.Tk()
    root.title("停車監控系統")
    
    def start_monitoring():
        messagebox.showinfo("資訊", "開始監控中...")
        detect_parking()
    
    tk.Label(root, text="車位監控系統", font=("Arial", 14)).pack(pady=10)
    tk.Button(root, text="開始監控", command=start_monitoring).pack(pady=5)
    tk.Button(root, text="退出", command=root.quit).pack(pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    start_gui()
