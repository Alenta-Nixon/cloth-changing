import sys
import cv2
import torch
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ultralytics import YOLO

from core.tracking_engine import update_tracks
from core.identity_database import IdentityDatabase
from core.feature_extractor import extract_features
from core.alert_engine import match_identity


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)


class CCTVSystem(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Cloth-Changing ReID Surveillance")
        self.resize(1700, 950)

        self.detector = YOLO("models/yolov8n.pt").to(device)

        self.db = IdentityDatabase()

        self.s_face = None
        self.s_app = None
        self.s_struct = None

        self.cap = None

        self.last_feature_time = 0
        self.feature_interval = 6

        self.setup_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)


# ------------------------------------------------
# UI
# ------------------------------------------------

    def setup_ui(self):

        layout = QHBoxLayout()

        left = QVBoxLayout()

        btn_video = QPushButton("Load Video")
        btn_video.clicked.connect(self.load_video)

        btn_susp = QPushButton("Upload Suspicious Image")
        btn_susp.clicked.connect(self.load_suspicious)

        left.addWidget(btn_video)
        left.addWidget(btn_susp)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(1100,700)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background:black; border:2px solid cyan;"
        )

        left.addWidget(self.video_label)
        layout.addLayout(left)

        right = QVBoxLayout()

        # ---------------------------
        # Suspicious person preview
        # ---------------------------
        right.addWidget(QLabel("Suspicious Person"))

        self.suspicious_preview = QLabel()
        self.suspicious_preview.setFixedSize(250,180)
        self.suspicious_preview.setAlignment(Qt.AlignCenter)
        self.suspicious_preview.setStyleSheet(
            "background:black; border:2px solid yellow;"
        )

        right.addWidget(self.suspicious_preview)

        # ---------------------------
        # Matched person preview
        # ---------------------------
        right.addWidget(QLabel("Matched Person"))

        self.thumb = QLabel()
        self.thumb.setFixedSize(250,180)
        self.thumb.setStyleSheet(
            "background:black; border:2px solid red;"
        )

        right.addWidget(self.thumb)

        # ---------------------------
        # Log panel
        # ---------------------------
        self.log = QListWidget()
        right.addWidget(self.log)

        layout.addLayout(right)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


# ------------------------------------------------
# Load Video
# ------------------------------------------------

    def load_video(self):

        path,_ = QFileDialog.getOpenFileName(
            self,"Video","","*.mp4 *.avi"
        )

        if path:
            self.cap = cv2.VideoCapture(path)
            self.log.addItem("Video loaded.")


# ------------------------------------------------
# Load Suspicious Image
# ------------------------------------------------

    def load_suspicious(self):

        path,_ = QFileDialog.getOpenFileName(
            self,"Image","","*.jpg *.png *.jpeg"
        )

        if path:

            img = cv2.imread(path)

            # display uploaded image
            rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            h,w,ch=rgb.shape
            img_qt=QImage(rgb.data,w,h,w*ch,QImage.Format_RGB888)
            pix=QPixmap.fromImage(img_qt)

            scaled=pix.scaled(
                self.suspicious_preview.width(),
                self.suspicious_preview.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.suspicious_preview.setPixmap(scaled)

            # extract identity features
            face, app, struct = extract_features(img,None)

            self.s_face = face
            self.s_app = app
            self.s_struct = struct

            self.log.addItem("Suspicious identity loaded.")


# ------------------------------------------------
# Frame Processing
# ------------------------------------------------

    def update_frame(self):

        if self.cap is None:
            return

        ret, frame = self.cap.read()

        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            return

        display = frame.copy()

        results = self.detector(frame,verbose=False)

        detections = []

        for box in results[0].boxes:

            if int(box.cls.item())==0:

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                detections.append(
                    ([x1,y1,x2-x1,y2-y1],1.0,"person")
                )

        tracks = update_tracks(detections,frame)

        current_time = cv2.getTickCount()/cv2.getTickFrequency()

        for track_id,x1,y1,x2,y2 in tracks:

            crop = frame[y1:y2,x1:x2]

            if crop is None or crop.size==0:
                continue

            if current_time - self.last_feature_time >= self.feature_interval:

                self.last_feature_time = current_time

                face,app,struct = extract_features(crop,None)

                self.db.update_identity(track_id,face,app,struct)

            mean_face,mean_app,mean_struct = self.db.get_mean(track_id)

            match_text="NO MATCH"
            color=(0,0,255)

            if self.s_face is not None:

                match,score = match_identity(
                    self.s_face,self.s_app,self.s_struct,
                    mean_face,mean_app,mean_struct
                )

                if match:

                    match_text=f"MATCH {score:.1f}%"
                    color=(0,255,0)

                    timestamp=datetime.now().strftime("%H:%M:%S")

                    self.log.addItem(
                        f"[{timestamp}] ID-{track_id} MATCH {score:.1f}%"
                    )

                    self.update_thumb(crop)

            cv2.rectangle(display,(x1,y1),(x2,y2),color,2)

            cv2.putText(
                display,
                f"ID-{track_id} {match_text}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,2
            )

        self.show_frame(display)


# ------------------------------------------------
# Display Video Frame
# ------------------------------------------------

    def show_frame(self,frame):

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        h,w,ch=rgb.shape

        img=QImage(rgb.data,w,h,w*ch,QImage.Format_RGB888)

        pix=QPixmap.fromImage(img)

        scaled=pix.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.video_label.setPixmap(scaled)


# ------------------------------------------------
# Update Matched Thumbnail
# ------------------------------------------------

    def update_thumb(self,crop):

        rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)

        h,w,ch=rgb.shape

        img=QImage(rgb.data,w,h,w*ch,QImage.Format_RGB888)

        pix=QPixmap.fromImage(img)

        scaled=pix.scaled(
            self.thumb.width(),
            self.thumb.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.thumb.setPixmap(scaled)


# ------------------------------------------------

if __name__=="__main__":

    app=QApplication(sys.argv)

    win=CCTVSystem()

    win.show()

    sys.exit(app.exec_())