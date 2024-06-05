import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QDesktopWidget, QComboBox, QHBoxLayout, QProgressBar
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont
import cv2
import numpy as np
from keras.models import model_from_json
import google.generativeai as genai

genai.configure(api_key="AIzaSyDrfSlMcKVzSDMscjrKOmzHzHk4spXuupM")

model_names = [
    'gemini-pro',
    'gemini-pro-vision'
]

filtered_models = [m for m in genai.list_models() if m.name in model_names]

for m in filtered_models:
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)

class OvalButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("border-radius: 20px;"
                           "background-color: #00CED1;"
                           "color: white;"
                           "padding: 15px 30px;"
                           "font-size: 24px;")

class AdviceThread(QThread):
    result_ready = pyqtSignal(str)
    
    def __init__(self, prediction_label, category):
        super().__init__()
        self.prediction_label = prediction_label
        self.category = category

    def run(self):
        try:
            model = genai.GenerativeModel('gemini-pro')
            query = f"{self.prediction_label} bir duyguya uygun sadece 1 tane {self.category} tavsiyesi ver."
            response = model.generate_content(query)
            advice_text = response.text
        except Exception as e:
            advice_text = f"Hata: {e}"
        self.result_ready.emit(advice_text)

class FacialEmotionAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Duygu Analizi")

        desktop = QApplication.desktop()
        screen_resolution = desktop.screenGeometry()
        screen_width, screen_height = screen_resolution.width(), screen_resolution.height()

        window_width, window_height = 1280, 720
        window_x = (screen_width - window_width) // 2
        window_y = (screen_height - window_height) // 2
        self.setGeometry(window_x, window_y, window_width, window_height)

        self.setWindowIcon(QIcon("logo.png"))

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.start_button = OvalButton('Analize Başla', self)
        self.start_button.clicked.connect(self.start_analysis)

        self.stop_button = OvalButton('Analizi Durdur', self)
        self.stop_button.clicked.connect(self.stop_analysis)

        self.advice_button = OvalButton('Tavsiye Al', self)
        self.advice_button.clicked.connect(self.get_advice)

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setFamily("Bahnschrift SemiBold")
        font.setPointSize(28)
        self.result_label.setFont(font)

        self.advice_label = QLabel(self)
        self.advice_label.setAlignment(Qt.AlignCenter)
        advice_font = QFont()
        advice_font.setFamily("Bahnschrift")
        advice_font.setPointSize(24)
        self.advice_label.setFont(advice_font)

        self.combo_box = QComboBox(self)
        self.combo_box.addItems(["Müzik", "Film", "Kitap", "Aktivite", "Dizi"])
        combo_font = QFont()
        combo_font.setFamily("Bahnschrift")
        combo_font.setPointSize(20)
        self.combo_box.setFont(combo_font)
        self.combo_box.setStyleSheet("background-color: #FFFFFF; padding: 5px;")
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(30, 40, 200, 25)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.combo_box)
        button_layout.addWidget(self.advice_button)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addLayout(button_layout)
        layout.addWidget(self.result_label, alignment=Qt.AlignHCenter)
        layout.addWidget(self.progress_bar, alignment=Qt.AlignHCenter)
        layout.addWidget(self.advice_label, alignment=Qt.AlignHCenter)

        self.central_widget.setLayout(layout)

        self.model = self.load_model("facialemotionmodel.json", "facialemotionmodel.h5")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.webcam = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.analyze_frame)
        #self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        self.labels = {0: 'sinirli', 1: 'iğrenme', 2: 'korku', 3: 'mutlu', 4: 'doğal', 5: 'üzgün', 6: 'şaşkın'}
        self.is_running = False
        self.prediction_label = None

    def load_model(self, model_json_file, model_weights_file):
        try:
            with open(model_json_file, "r") as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json)
            model.load_weights(model_weights_file)
            return model
        except Exception as e:
            print(f"Model yükleme hatasi: {e}")
            sys.exit(1)

    def start_analysis(self):
        if not self.is_running:
            if not self.webcam.isOpened():
                print("Webcam açilamadi!")
                return
            self.timer.start(100)
            self.is_running = True

    def stop_analysis(self):
        if self.is_running:
            self.timer.stop()
            self.is_running = False

    def analyze_frame(self):
        try:
            ret, frame = self.webcam.read()
            if not ret:
                print("Kare okunamadi!")
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)
            for (x, y, w, h) in faces:
                face_image = frame[y:y+h, x:x+w]
                face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
                face_image = cv2.resize(face_image, (48, 48))
                face_image = np.array(face_image)
                face_image = face_image.reshape(1, 48, 48, 1) / 255.0
                pred = self.model.predict(face_image)
                self.prediction_label = self.labels[pred.argmax()]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.video_label.setPixmap(pixmap)
            self.result_label.setText(self.prediction_label)
        except Exception as e:
            print(f"Kare analizi hatasi: {e}")

    def get_advice(self):
        if self.prediction_label:
            category = self.combo_box.currentText().lower()
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            self.advice_thread = AdviceThread(self.prediction_label, category)
            self.advice_thread.result_ready.connect(self.display_advice)
            self.advice_thread.start()
            
            self.timer_progress = QTimer()
            self.timer_progress.timeout.connect(self.update_progress_bar)
            self.timer_progress.start(50)
        else:
            self.advice_label.setText("Önce yüz ifadesi analizi yapin.")

    def update_progress_bar(self):
        current_value = self.progress_bar.value()
        if current_value < 100:
            self.progress_bar.setValue(current_value + 5)
        else:
            self.timer_progress.stop()

    def display_advice(self, advice_text):
        self.progress_bar.setVisible(False)
        self.advice_label.setText(advice_text)

    def closeEvent(self, event):
        self.webcam.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FacialEmotionAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())