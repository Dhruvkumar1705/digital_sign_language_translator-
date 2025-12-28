import cv2
import tkinter as tk
from tkinter import ttk
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import time
import pyttsx3
from PIL import Image, ImageTk
from collections import deque

SUGGESTION_WORDS = [
    # Greetings
    "HELLO", "HI", "GOOD MORNING", "GOOD AFTERNOON", "GOOD EVENING", "GOOD NIGHT",
    "WELCOME", "GOODBYE", "BYE", "SEE YOU", "SEE YOU LATER", "TAKE CARE",
    
    # Common phrases
    "THANK YOU", "THANKS", "THANK", "PLEASE", "SORRY", "EXCUSE ME",
    "YOU ARE WELCOME", "NO PROBLEM", "MY PLEASURE",
    
    # Questions
    "WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHO", "WHICH",
    "WHAT IS YOUR NAME", "HOW ARE YOU", "WHERE ARE YOU FROM",
    "WHAT TIME", "HOW MUCH", "CAN YOU HELP",
    
    # Pronouns
    "I", "YOU", "HE", "SHE", "WE", "THEY", "ME", "US", "THEM",
    "MY", "YOUR", "HIS", "HER", "OUR", "THEIR",
    
    # Basic needs
    "HELP", "NEED", "WANT", "LIKE", "LOVE", "HATE",
    "EAT", "DRINK", "SLEEP", "REST", "BATHROOM", "WATER", "FOOD",
    
    # Feelings
    "HAPPY", "SAD", "ANGRY", "TIRED", "SICK", "FINE", "GOOD", "BAD",
    "EXCITED", "WORRIED", "SCARED", "CALM", "SORRY",
    
    # Yes/No
    "YES", "NO", "MAYBE", "OKAY", "SURE", "ALRIGHT",
    
    # Family
    "FAMILY", "MOTHER", "FATHER", "BROTHER", "SISTER", "FRIEND",
    "CHILD", "BABY", "GRANDMOTHER", "GRANDFATHER",
    
    # Places
    "HOME", "HOUSE", "SCHOOL", "WORK", "HOSPITAL", "STORE",
    "RESTAURANT", "LIBRARY", "PARK", "CITY",
    
    # Time
    "TODAY", "TOMORROW", "YESTERDAY", "NOW", "LATER", "SOON",
    "MORNING", "AFTERNOON", "EVENING", "NIGHT", "DAY", "WEEK",
    
    # Numbers
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN",
    
    # Actions
    "GO", "COME", "STOP", "START", "FINISH", "WAIT", "RUN", "WALK",
    "SIT", "STAND", "GIVE", "TAKE", "MAKE", "DO", "SAY", "TELL",
    
    # Common adjectives
    "BIG", "SMALL", "NEW", "OLD", "HOT", "COLD", "EASY", "HARD",
    "FAST", "SLOW", "NEAR", "FAR", "RIGHT", "WRONG", "SAME", "DIFFERENT",
    
    # Emergency
    "EMERGENCY", "DANGER", "CALL", "POLICE", "AMBULANCE", "FIRE"
]

# ===================== ASL TRANSLATOR =====================

class ASLTranslator:
    def __init__(self, model_path, class_mapping_path, confidence_threshold=0.7):

        custom_objects = {'TransformerBlock': self.TransformerBlock}
        self.model = tf.keras.models.load_model(
            model_path, custom_objects=custom_objects, compile=False
        )

        with open(class_mapping_path) as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.7
        )

        self.confidence_threshold = confidence_threshold

        self.predicted_char = None
        self.prediction_confidence = 0
        self.prediction_buffer = deque(maxlen=15)
        self.stable_char = None

        self.current_sentence = []
        self.current_buffer = ""
        self.last_time = time.time()
        self.pause_time = 1.2

        self.top_predictions = []

        self.suggestions = []

        self.tts_engine = pyttsx3.init()

    class TransformerBlock(tf.keras.layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim):
            super().__init__()
            self.att = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads
            )
            self.ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ])
            self.ln1 = tf.keras.layers.LayerNormalization()
            self.ln2 = tf.keras.layers.LayerNormalization()

        def call(self, x):
            attn = self.att(x, x)
            x = self.ln1(x + attn)
            ffn = self.ffn(x)
            return self.ln2(x + ffn)

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, self.mp_hands.HAND_CONNECTIONS
            )

            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).reshape(1, 1, -1)
            preds = self.model.predict(landmarks, verbose=0)[0]

            top_idx = np.argsort(preds)[-3:][::-1]
            self.top_predictions = [
                (self.idx_to_class[i], float(preds[i])) for i in top_idx
            ]

            char, conf = self.top_predictions[0]

            if conf >= self.confidence_threshold:
                self.prediction_buffer.append(char)

                if self.prediction_buffer.count(char) > 10:
                    self.predicted_char = char
                    self.prediction_confidence = conf
                    self.current_buffer = char
                    self.stable_char = char
                    self.last_time = time.time()

        self._update_sentence()
        self.update_suggestions()

        return frame

    def _update_sentence(self):
        if self.stable_char and (time.time() - self.last_time) > self.pause_time:
            self.current_sentence.append(self.stable_char)
            self.stable_char = None
            self.current_buffer = ""

    def get_prediction(self):
        return self.predicted_char, self.prediction_confidence

    def update_suggestions(self):
        """Generate word suggestions based on current buffer"""
        if not self.current_buffer:
            self.suggestions = []
            return

        prefix = self.current_buffer.upper()
        self.suggestions = [
            word for word in SUGGESTION_WORDS
            if word.startswith(prefix)
        ][:5]  # max 5 suggestions


# ===================== GUI APP =====================

class ASLApp:
    def __init__(self, root):
        self.root = root
        root.title("ASL Sign Language Translator")
        root.geometry("1100x700")

        self.translator = ASLTranslator(
            "models/model_20251226_132123/best_model.h5",
            "models/model_20251226_132123/class_mapping.json"
        )

        self.main = ttk.Frame(root, padding=10)
        self.main.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(self.main)
        self.video_label.pack(side=tk.LEFT)

        self.panel = ttk.Frame(self.main, width=350)
        self.panel.pack(side=tk.RIGHT, fill=tk.Y)

        self.pred_label = ttk.Label(
            self.panel, text="Show Sign", font=("Helvetica", 26, "bold")
        )
        self.pred_label.pack(pady=10)

        self.conf_label = ttk.Label(self.panel, text="Confidence: 0%")
        self.conf_label.pack()

        self.conf_bar = ttk.Progressbar(self.panel, length=200, maximum=100)
        self.conf_bar.pack(pady=5)

        ttk.Label(self.panel, text="Top Predictions", font=("Helvetica", 12, "bold")).pack()
        self.top_box = tk.Text(self.panel, height=4, width=30, state="disabled")
        self.top_box.pack()

        ttk.Label(self.panel, text="Suggestions", font=("Helvetica", 12, "bold")).pack(pady=(10, 0))
        self.suggestion_box = tk.Text(self.panel, height=4, width=30, state="disabled")
        self.suggestion_box.pack()

        ttk.Label(self.panel, text="Buffer", font=("Helvetica", 12, "bold")).pack(pady=5)
        self.buffer_label = ttk.Label(self.panel, relief="solid", width=30)
        self.buffer_label.pack()

        ttk.Label(self.panel, text="Sentence", font=("Helvetica", 12, "bold")).pack(pady=5)
        self.sentence_label = ttk.Label(
            self.panel, relief="solid", wraplength=300
        )
        self.sentence_label.pack()

        self.speak_btn = ttk.Button(
            self.panel, text="🔊 Speak Sentence", command=self.speak
        )
        self.speak_btn.pack(pady=5)

        self.space_btn = ttk.Button(
            self.panel, text="␣ Add Space", command=self.add_space
        )
        self.space_btn.pack()

        self.fps_label = ttk.Label(self.panel, text="FPS: 0")
        self.fps_label.pack(pady=5)

        self.cap = cv2.VideoCapture(0)
        self.last_fps_time = time.time()

        self.update()

    def speak(self):
        text = " ".join(self.translator.current_sentence)
        if text:
            self.translator.tts_engine.say(text)
            self.translator.tts_engine.runAndWait()

    def add_space(self):
        self.translator.current_sentence.append("")

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            frame = self.translator.process_frame(frame)

            char, conf = self.translator.get_prediction()
            if char:
                self.pred_label.config(text=char)
                self.conf_label.config(text=f"Confidence: {conf:.1%}")
                self.conf_bar["value"] = conf * 100

            self.buffer_label.config(text=self.translator.current_buffer)
            self.sentence_label.config(
                text=" ".join(self.translator.current_sentence)
            )

            self.top_box.config(state="normal")
            self.top_box.delete("1.0", tk.END)
            for c, p in self.translator.top_predictions:
                self.top_box.insert(tk.END, f"{c} : {p:.2%}\n")
            self.top_box.config(state="disabled")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_label.imgtk = img
            self.video_label.config(image=img)

            self.suggestion_box.config(state="normal")
            self.suggestion_box.delete("1.0", tk.END)
            
            for word in self.translator.suggestions:
                self.suggestion_box.insert(tk.END, f"{word}\n")

            self.suggestion_box.config(state="disabled")


        now = time.time()
        fps = 1 / (now - self.last_fps_time)
        self.last_fps_time = now
        self.fps_label.config(text=f"FPS: {int(fps)}")

        self.root.after(10, self.update)

# ===================== RUN =====================

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLApp(root)
    root.mainloop()
