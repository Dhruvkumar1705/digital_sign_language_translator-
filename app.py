from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import pyttsx3
from collections import deque
import threading
import base64
from deep_translator import GoogleTranslator

app = Flask(__name__)

# ================= ENHANCED ASL TRANSLATOR =================

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

    # ===== 🔥 MOST USED DAILY (TOP PRIORITY) =====
"I","YOU","ME","MY","YOUR","WE","US","THEY","THEM",
"YES","NO","OK","OKAY","PLEASE","THANK YOU","THANKS","SORRY",
"HELP","NEED","WANT","LIKE","LOVE","HATE",
"GOOD","BAD","FINE","BETTER","BEST",
"HELLO","HI","BYE","WELCOME",
"GO","COME","STOP","WAIT","START","FINISH",
"EAT","DRINK","WATER","FOOD","HUNGRY","THIRSTY",
"SLEEP","REST","TIRED","SICK","PAIN",
"NOW","LATER","TODAY","TOMORROW","YESTERDAY",

# ===== QUESTIONS =====
"WHAT","WHERE","WHEN","WHY","HOW","WHO","WHICH",
"HOW MUCH","HOW MANY","WHAT TIME",
"CAN YOU HELP","DO YOU UNDERSTAND","IS THIS OK",

# ===== FEELINGS =====
"HAPPY","SAD","ANGRY","EXCITED","SCARED","WORRIED",
"CALM","RELAX","CONFUSED","PROUD","ASHAMED",

# ===== FAMILY =====
"MOTHER","FATHER","BROTHER","SISTER",
"FRIEND","FAMILY","CHILD","BABY","HUSBAND","WIFE",

# ===== PLACES =====
"HOME","HOUSE","SCHOOL","COLLEGE","OFFICE","WORK",
"HOSPITAL","SHOP","MARKET","RESTAURANT","PARK","ROAD",

# ===== TIME =====
"MORNING","AFTERNOON","EVENING","NIGHT",
"DAY","WEEK","MONTH","YEAR",

# ===== EMERGENCY =====
"EMERGENCY","DANGER","HELP ME",
"CALL POLICE","CALL AMBULANCE","FIRE","ACCIDENT",

# ===== A =====
"ABLE","ABOUT","ACCEPT","AGAIN","ALWAYS","ALONE","ANSWER","ARRIVE","ASK",

# ===== B =====
"BACK","BAD","BECAUSE","BEGIN","BEFORE","BELIEVE","BUSY","BUY","BRING",

# ===== C =====
"CALL","CARE","CHANGE","CHECK","CLOSE","COME","COMPLETE","CORRECT",

# ===== D =====
"DONE","DOOR","DOWN","DRINK","DRIVE","DIFFERENT","DOUBT",

# ===== E =====
"EASY","EAT","END","ENOUGH","EXPLAIN","EXTRA","EARLY",

# ===== F =====
"FAST","FEEL","FIND","FINISH","FOLLOW","FREE","FORGET","FORWARD",

# ===== G =====
"GET","GIVE","GO","GOOD","GREAT","GROUP","GUESS",

# ===== H =====
"HAPPY","HARD","HEAR","HELP","HERE","HIGH","HOLD","HOPE",

# ===== I =====
"IDEA","IMPORTANT","INSIDE","INVITE","ISSUE",

# ===== J =====
"JOIN","JOB","JUST","JUDGE",

# ===== K =====
"KEEP","KNOW","KIND","KITCHEN",

# ===== L =====
"LATE","LEARN","LEAVE","LEFT","LISTEN","LOOK","LOSE",

# ===== M =====
"MAKE","MEET","MORE","MOVE","MONEY","MISTAKE",

# ===== N =====
"NEAR","NEED","NEXT","NICE","NORMAL","NOW",

# ===== O =====
"OPEN","ORDER","OUT","ONLY","OVER",

# ===== P =====
"PAY","PEOPLE","PLEASE","PROBLEM","PUT","PLAN",

# ===== Q =====
"QUESTION","QUICK","QUIET",

# ===== R =====
"READY","READ","REST","RIGHT","RUN","REMEMBER",

# ===== S =====
"SAY","SEE","SIT","SLOW","START","STOP","STUDY","SAFE",

# ===== T =====
"TAKE","TALK","THINK","TIME","TRY","TRUST",

# ===== U =====
"UNDERSTAND","USE","USUAL",

# ===== V =====
"VERY","VISIT","VOICE",

# ===== W =====
"WALK","WAIT","WANT","WORK","WATCH","WRONG",

# ===== X =====
"EXPLAIN",

# ===== Y =====
"YES","YESTERDAY","YOUNG",

# ===== Z =====
"ZERO"
]

# ================= OPTIMIZATIONS =================
class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.started = False
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            ret, frame = self.capture.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.01) # Small sleep to preventing CPU hogging

    def read(self):
        with self.lock:
            return self.ret, self.frame if self.frame is not None else None

    def stop(self):
        self.started = False
        self.thread.join()
        self.capture.release()

# Global camera instance
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = ThreadedCamera().start()
        # Wait for camera to warm up
        time.sleep(1.0) 
    return camera


class ASLTranslator:
    def __init__(self, model_path, class_mapping_path, confidence_threshold=0.65):
        # Load model with custom TransformerBlock
        custom_objects = {'TransformerBlock': self.TransformerBlock}
        self.model = tf.keras.models.load_model(
            model_path, custom_objects=custom_objects, compile=False
        )

        with open(class_mapping_path) as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Prediction settings
        self.confidence_threshold = confidence_threshold
        self.predicted_char = None
        self.prediction_confidence = 0
        self.prediction_buffer = deque(maxlen=15)
        self.stable_char = None
        
        # Sentence building
        self.current_sentence = []
        self.current_buffer = ""
        self.last_time = time.time()
        self.pause_time = 0.8 # Reduced from 1.2 for faster typing
        self.word_buffer = ""   # 🔥 NEW: stores full word before committing
        self.word_pause_time = 1.0  # Reduced from 1.5

        # Additional features
        self.top_predictions = []
        self.suggestions = []

        # Text-to-speech (initialized lazily to avoid shutdown/destructor issues)
        self.tts_engine = None

        # FPS Calculation
        self.fps = 0
        self.prev_frame_time = 0
        
        # Action Manager State
        self.last_action = None

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

    def preprocess_frame(self, frame):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to improve contrast in varying lighting conditions (e.g. sunlight).
        """
        try:
            # 1. Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 2. Apply CLAHE to L-channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # 3. Merge and convert back to BGR
            limg = cv2.merge((cl,a,b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception as e:
            print(f"Preprocessing Error: {e}")
            return frame

    def process_frame(self, frame):
        """Process frame and extract hand landmarks for prediction"""
        # Apply preprocessing for lighting robustness
        frame = self.preprocess_frame(frame)

        # Calculate FPS
        new_frame_time = time.time()
        time_diff = new_frame_time - self.prev_frame_time
        if self.prev_frame_time > 0 and time_diff > 0:
            self.fps = 1 / time_diff
        self.prev_frame_time = new_frame_time

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mp_hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            
            # Draw hand landmarks
            self.mp_drawing.draw_landmarks(
                frame, 
                hand, 
                mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )

            # Extract landmarks
            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).reshape(1, 1, -1)
            
            # OPTIMIZATION: Call model directly instead of .predict()
            # .predict() has overhead for batching which we don't need
            preds = self.model(landmarks, training=False).numpy()[0]

            # Get top 3 predictions
            top_idx = np.argsort(preds)[-3:][::-1]
            self.top_predictions = [
                (self.idx_to_class[i], float(preds[i])) for i in top_idx
            ]
            # Filter removed to keep UI stable with 3 predictions always
            # self.top_predictions = [p for p in self.top_predictions if p[1] > 0.05]

            char, conf = self.top_predictions[0] if self.top_predictions else (None, 0)

            # Update prediction buffer
            if char and conf >= self.confidence_threshold:
                self.prediction_buffer.append(char)

                # Check if prediction is stable
                # OPTIMIZATION: Reduced from 8/15 to 5/10 for responsiveness
                stable_count = max(5, int(10 * conf))
                if self.prediction_buffer.count(char) >= stable_count:
                    self.predicted_char = char
                    self.prediction_confidence = conf
                    self.current_buffer = char
                    self.stable_char = char
                    self.last_time = time.time()

        # Update sentence and suggestions
        self._update_sentence()
        self.update_suggestions()
        
        return frame

    def _update_sentence(self):
        """
        Commit characters into a word buffer.
        Auto-commit REMOVED to emulate keyboard behavior.
        Words are committed only via 'Space' or 'Select Suggestion'.
        """
        if self.stable_char and (time.time() - self.last_time) > self.pause_time:
            # Avoid duplicate letters if it's the same as the last committed char
            if not self.word_buffer or self.word_buffer[-1] != self.stable_char:
                self.word_buffer += self.stable_char
            
            self.stable_char = None
            self.current_buffer = ""
            self.last_time = time.time()

        # REMOVED: Auto-commit of full word logic

    def update_suggestions(self):
        # Use word_buffer (accumulated chars) for suggestions

        # Strict prefix matching
        if not self.word_buffer:
            self.suggestions = []
            return

        # ================= ACTION MANAGER LOGIC =================
        word = self.word_buffer.upper()
        if word in ["TIME", "DATE", "THEME"]:
            self.last_action = word
            self.word_buffer = "" # Clear buffer immediately
            self.current_buffer = ""
            return
        # ========================================================

        prefix = self.word_buffer.upper()

        self.suggestions = [
            word for word in SUGGESTION_WORDS
            if word.startswith(prefix)
        ][:5]


    def get_state(self):
        """Get current translator state"""
        # Display: Sentence + Current forming word (buffer)
        sentence_text = ''.join(self.current_sentence) + self.word_buffer

        return {
            'predicted_char': self.predicted_char or '-',
            'confidence': float(self.prediction_confidence) if self.prediction_confidence else 0,
            'current_buffer': self.current_buffer or '-', # Shows immediate detection
            'sentence': sentence_text or 'Start signing to see translation...',
            'top_predictions': self.top_predictions,
            'suggestions': self.suggestions,
            'fps': int(self.fps),
            'action': self.last_action # Return triggered action
        }
        
    def reset_action(self):
        self.last_action = None

    def add_space(self):
        """Add space to current sentence"""
        # 1. If we have a word in buffer, commit it first
        if self.word_buffer:
            self.current_sentence.append(self.word_buffer)
            self.word_buffer = ""
        
        # 2. Add the space if not already present
        if self.current_sentence and self.current_sentence[-1] != ' ':
             self.current_sentence.append(' ')

        # Flush any pending stable character
        self.stable_char = None
        self.current_buffer = ""

    def clear_sentence(self):
        """Clear current sentence"""
        self.current_sentence = []
        self.word_buffer = "" # Clear buffer too
        self.current_buffer = ""
        self.predicted_char = None
        self.prediction_confidence = 0
        self.stable_char = None
    
    def backspace(self):
        """Remove last character/word"""
        # 1. Delete from word_buffer first (if typing a word)
        if self.word_buffer:
            self.word_buffer = self.word_buffer[:-1]
        
        # 2. If buffer empty, delete from sentence
        elif self.current_sentence:
            self.current_sentence.pop()
            
        self.current_buffer = ""
        # Recalculate suggestions based on new buffer state
        self.update_suggestions() 

    def speak_sentence(self):
        """Speak the current sentence using TTS"""
        # Include buffer in speech if desired, or just committed sentence
        full_text = ''.join(self.current_sentence) + self.word_buffer
        text = full_text.strip()
        if not text:
            return

        t = threading.Thread(target=self._speak, args=(text,), daemon=True)
        t.start()

    def _create_tts_engine(self):
        try:
            return pyttsx3.init()
        except Exception as e:
            print(f"TTS init failed: {e}")
            return None

    def _speak(self, text):
        """Internal TTS method"""
        engine = self._create_tts_engine()
        if engine is None:
            return

        try:
            engine.setProperty('rate', 140)  # Slower speed (default is usually ~200)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            try:
                engine.stop()
            except Exception:
                pass
            del engine

    def select_suggestion(self, word):
        """Select a suggested word"""
        # 1. Commit the selected word
        self.current_sentence.append(word)
        self.current_sentence.append(' ')
        
        # 2. RESET buffer completely
        self.word_buffer = ""
        self.current_buffer = ""
        
        # 3. Clear suggestions immediately
        self.suggestions = []

# ================= INITIALIZE MODEL =================
translator = ASLTranslator(
    "models/model_20260106_133622/best_model.h5",
    "models/model_20260106_133622/class_mapping.json"
)

# ================= ROUTES =================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            cam = camera # Get global camera instance
            
            if cam is None:
                # Camera is stopped, yield placeholder or sleep
                time.sleep(0.5)
                continue

            # Non-blocking read
            ret, frame = cam.read()
            if not ret or frame is None:
                # If no frame yet, wait a bit
                time.sleep(0.01)
                continue

            # 🔥 ML inference here (Optimized/Safe)
            # Copy frame to avoid thread contention if any
            frame_copy = frame.copy()
            frame_processed = translator.process_frame(frame_copy)

            ret, jpeg = cv2.imencode(".jpg", frame_processed)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg.tobytes() +
                b"\r\n"
            )

            # Cap FPS to avoid sending too fast if processing is super fast
            time.sleep(0.005)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/state")
def state():
    state = translator.get_state()
    translator.reset_action() # Consume the action event
    return jsonify(state)


@app.route("/speak")
def speak():
    translator.speak_sentence()
    return jsonify(success=True)


@app.route("/space")
def space():
    translator.add_space()
    return jsonify(success=True)


@app.route("/backspace")
def backspace():
    translator.backspace()
    return jsonify(success=True)


@app.route("/clear")
def clear():
    translator.clear_sentence()
    return jsonify(success=True)

@app.route("/select/<word>")
def select(word):
    translator.select_suggestion(word)
    return jsonify(success=True)


@app.route("/process_speech", methods=["POST"])
def process_speech():
    data = request.json
    text = data.get("text", "").lower().strip()
    
    if not text:
        return jsonify(sequence=[])

    isl_gifs_dir = os.path.join(app.root_path, "static", "isl_gifs")
    
    # Get all available GIF files
    available_gifs = {}
    if os.path.exists(isl_gifs_dir):
        for filename in os.listdir(isl_gifs_dir):
            if filename.lower().endswith('.gif'):
                # Normalize filename to phrase: 'thank-you.gif' -> 'thank you', 'good_morning.gif' -> 'good morning'
                name_part = os.path.splitext(filename)[0].lower()
                phrase = name_part.replace('-', ' ').replace('_', ' ')
                available_gifs[phrase] = filename

    # Simple Greedy Matching
    # Split text into words
    import re
    words = re.findall(r'\b\w+\b', text)
    
    found_sequence = []
    i = 0
    while i < len(words):
        match_found = False
        # Try to match phrases starting from max length (e.g. 3 words) down to 1
        for length in range(3, 0, -1):
            if i + length <= len(words):
                phrase = " ".join(words[i:i+length])
                if phrase in available_gifs:
                    found_sequence.append({
                        "word": phrase,
                        "type": "gif",
                        "src": f"/static/isl_gifs/{available_gifs[phrase]}"
                    })
                    i += length
                    match_found = True
                    break
        
        if not match_found:
            # No GIF found for word[i]
            found_sequence.append({
                "word": words[i],
                "type": "text",
                "src": None
            })
            i += 1
            
    return jsonify(sequence=found_sequence)


@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.json
        text = data.get("text", "")
        target_lang = data.get("target", "hi")
        
        if not text:
            return jsonify(translated_text="")
            
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return jsonify(translated_text=translated)
    except Exception as e:
        print(f"Translation error: {e}")
        return jsonify(error=str(e)), 500


@app.route("/stop_camera")
def stop_camera():
    global camera
    if camera:
        camera.stop()
        camera = None
    return jsonify(success=True)


@app.route("/start_camera")
def start_camera():
    global camera
    if camera is None:
        camera = ThreadedCamera().start()
        time.sleep(1.0) # Warmup
    return jsonify(success=True)




if __name__ == "__main__":
    # Threaded=True might help flask too, but we are managing our own threads
   app.run(debug=False, use_reloader=False, threaded=True)