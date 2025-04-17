import cv2
import mediapipe as mp
import pygame
import threading
import time
from tkinter import *
from tkinter import filedialog

# 音楽初期化
pygame.mixer.init()

# MediaPipeの顔検出
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

# GUIと状態管理
class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("居眠り検知アプリ")
        self.root.geometry("400x300")
        
        self.music_file = None
        self.monitoring = False

        self.label = Label(root, text="音楽ファイルを選択してください")
        self.label.pack(pady=10)

        self.select_button = Button(root, text="音楽を選ぶ", command=self.select_music)
        self.select_button.pack(pady=5)

        self.start_button = Button(root, text="監視開始", command=self.start_monitoring)
        self.start_button.pack(pady=10)

        self.stop_button = Button(root, text="監視停止", command=self.stop_monitoring, state=DISABLED)
        self.stop_button.pack(pady=5)

        self.status = Label(root, text="ステータス: 待機中")
        self.status.pack(pady=20)

    def select_music(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav")])
        if file_path:
            self.music_file = file_path
            self.label.config(text=f"選択された音楽: {file_path.split('/')[-1]}")
    
    def start_monitoring(self):
        if not self.music_file:
            self.status.config(text="ステータス: 音楽を選択してください")
            return
        self.monitoring = True
        self.start_button.config(state=DISABLED)
        self.stop_button.config(state=NORMAL)
        self.status.config(text="ステータス: 監視中...")
        threading.Thread(target=self.monitor).start()

    def stop_monitoring(self):
        self.monitoring = False
        self.start_button.config(state=NORMAL)
        self.stop_button.config(state=DISABLED)
        self.status.config(text="ステータス: 停止中")

    def play_alarm(self):
        pygame.mixer.music.load(self.music_file)
        pygame.mixer.music.set_volume(1.0)  # 最大音量
        pygame.mixer.music.play()

    def monitor(self):
        cap = cv2.VideoCapture(0)
        eye_closed_frames = 0
        EYE_AR_THRESHOLD = 0.3
        CONSEC_FRAMES = 30  # 約1秒あたり30フレーム想定

        while self.monitoring and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            if result.multi_face_landmarks:
                for landmarks in result.multi_face_landmarks:
                    left_eye = [landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
                    right_eye = [landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]

                    def eye_aspect_ratio(eye):
                        A = ((eye[1].x - eye[5].x)**2 + (eye[1].y - eye[5].y)**2)**0.5
                        B = ((eye[2].x - eye[4].x)**2 + (eye[2].y - eye[4].y)**2)**0.5
                        C = ((eye[0].x - eye[3].x)**2 + (eye[0].y - eye[3].y)**2)**0.5
                        return (A + B) / (2.0 * C)

                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    avg_ear = (left_ear + right_ear) / 2

                    if avg_ear < EYE_AR_THRESHOLD:
                        eye_closed_frames += 1
                    else:
                        eye_closed_frames = 0

                    if eye_closed_frames > CONSEC_FRAMES * 2:  # 約2秒
                        self.status.config(text="ステータス: 居眠り検知！")
                        self.play_alarm()
                        eye_closed_frames = 0

            time.sleep(0.03)  # 約30FPS
        cap.release()

# 実行
if __name__ == "__main__":
    root = Tk()
    app = DrowsinessApp(root)
    root.mainloop()
