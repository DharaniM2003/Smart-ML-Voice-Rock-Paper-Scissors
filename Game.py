import gradio as gr
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pyttsx3
import threading
import time
from collections import deque

# Speech Recognition
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("⚠️ speech_recognition not installed. Voice input disabled.")

# Optional VLC for streaming
try:
    import vlc
    VLC_AVAILABLE = True
except ImportError:
    VLC_AVAILABLE = False
    print("⚠️ python-vlc not installed. Music streaming disabled.")

# =====================================
# 🎵 MUSIC STREAMING (VLC)
# =====================================
class MusicStreamer:
    def __init__(self, url):
        self.url = url
        self.instance = None
        self.player = None
        self.is_playing = False
        
        if not VLC_AVAILABLE:
            return
            
        try:
            self.instance = vlc.Instance('--quiet')
            self.player = self.instance.media_player_new()
            media = self.instance.media_new(self.url)
            self.player.set_media(media)
            self.player.audio_set_volume(60)
        except Exception as e:
            print(f"VLC initialization error: {e}")
            self.instance = None
            self.player = None

    def play(self):
        if self.player and not self.is_playing:
            try:
                self.player.play()
                self.is_playing = True
                return True
            except Exception as e:
                print(f"Music play error: {e}")
                return False
        return False

    def pause(self):
        if self.player and self.is_playing:
            self.player.pause()
            self.is_playing = False

    def stop(self):
        if self.player:
            try:
                self.player.stop()
                self.is_playing = False
            except:
                pass

# =====================================
# 🔊 TEXT-TO-SPEECH
# =====================================
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 165)
    tts_engine.setProperty("volume", 0.9)
    TTS_AVAILABLE = True
except Exception as e:
    print(f"⚠️ TTS not available: {e}")
    TTS_AVAILABLE = False

def speak(text):
    if not TTS_AVAILABLE:
        return
    def _speak():
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
    threading.Thread(target=_speak, daemon=True).start()

# =====================================
# 🎤 SPEECH RECOGNITION
# =====================================
if SR_AVAILABLE:
    recognizer = sr.Recognizer()

def listen_for_voice():
    """Listen to microphone and recognize Rock/Paper/Scissors"""
    if not SR_AVAILABLE:
        return None, "❌ Speech recognition not installed. Run: pip install SpeechRecognition pyaudio"
    
    try:
        with sr.Microphone() as source:
            speak("Listening... Say Rock, Paper, or Scissors")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            status_msg = "🎤 Listening... Speak now!"
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
            
            status_msg = "🔄 Processing your speech..."
            text = recognizer.recognize_google(audio).lower()
            
            print(f"Heard: {text}")
            
            # Check for valid choice
            for choice in CHOICES:
                if choice.lower() in text:
                    speak(f"You said {choice}")
                    return choice, f"✅ Recognized: {choice}"
            
            speak("I didn't catch Rock, Paper or Scissors. Try again.")
            return None, f"❓ Heard '{text}' but couldn't find valid move. Try again!"
            
    except sr.WaitTimeoutError:
        return None, "⏱️ Timeout - No speech detected. Click again to try."
    except sr.UnknownValueError:
        return None, "❓ Couldn't understand audio. Speak clearly and try again."
    except sr.RequestError as e:
        return None, f"🌐 Speech API error: {e}"
    except Exception as e:
        return None, f"❌ Microphone error: {e}. Check if mic is connected."

# =====================================
# 🧠 ML MODEL + GAME LOGIC
# =====================================
CHOICES = ['Rock', 'Paper', 'Scissors']
MIN_DATA_FOR_ML = 5
LEVEL_UP_EVERY = 3

# ML Training Data
history_user = deque(maxlen=100)  # User's last moves
history_comp = deque(maxlen=100)  # Computer's responses
ml_model = DecisionTreeClassifier(max_depth=5, random_state=42)

# Game State
game_state = {
    "user_score": 0,
    "comp_score": 0,
    "draws": 0,
    "round": 0,
    "level": 1,
    "streak": 0,
    "history": []
}

# Image URLs (hosted online) - Using reliable sources
IMAGE_URLS = {
    "Rock": "https://em-content.zobj.net/thumbs/240/apple/354/rock_1faa8.png",
    "Paper": "https://em-content.zobj.net/thumbs/240/apple/354/rolled-up-newspaper_1f5de-fe0f.png",
    "Scissors": "https://em-content.zobj.net/thumbs/240/apple/354/scissors_2702-fe0f.png",
    "Blank": "https://em-content.zobj.net/thumbs/240/apple/354/thinking-face_1f914.png"
}

# Initialize music
MUSIC_URL = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
music_player = MusicStreamer(MUSIC_URL) if VLC_AVAILABLE else None

# =====================================
# 🎮 CORE GAME LOGIC
# =====================================
def predict_computer_move(user_choice_idx):
    """Use ML to predict user's next move and counter it"""
    if len(history_user) < MIN_DATA_FOR_ML:
        return random.randint(0, 2)
    
    try:
        # Train on historical data
        X = np.array(list(history_user)).reshape(-1, 1)
        y = np.array(list(history_comp))
        ml_model.fit(X, y)
        
        # Predict what user might play next and counter it
        predicted_user = ml_model.predict([[user_choice_idx]])[0]
        # Counter: Rock(0)->Paper(1), Paper(1)->Scissors(2), Scissors(2)->Rock(0)
        counter_move = (predicted_user + 1) % 3
        return counter_move
    except Exception as e:
        print(f"ML prediction error: {e}")
        return random.randint(0, 2)

def determine_winner(user_idx, comp_idx):
    """Returns: 'user', 'computer', or 'draw'"""
    if user_idx == comp_idx:
        return 'draw'
    elif (user_idx == 0 and comp_idx == 2) or \
         (user_idx == 1 and comp_idx == 0) or \
         (user_idx == 2 and comp_idx == 1):
        return 'user'
    else:
        return 'computer'

def play_round(user_choice):
    """Main game round logic - automatically plays when user selects"""
    if not user_choice:
        return (
            "❌ Please select Rock, Paper, or Scissors first!",
            IMAGE_URLS["Blank"],
            IMAGE_URLS["Blank"],
            format_scoreboard(),
            format_history()
        )
    
    # Small delay for better UX
    time.sleep(0.3)
    
    user_idx = CHOICES.index(user_choice)
    
    # AI Decision
    comp_idx = predict_computer_move(user_idx)
    comp_choice = CHOICES[comp_idx]
    
    # Record history for ML
    history_user.append(user_idx)
    history_comp.append(comp_idx)
    
    # Determine winner
    winner = determine_winner(user_idx, comp_idx)
    
    # Update scores
    game_state["round"] += 1
    result_msg = ""
    
    if winner == 'user':
        game_state["user_score"] += 1
        game_state["streak"] += 1
        result_msg = "🎉 YOU WIN!"
        speak(f"You win with {user_choice}!")
    elif winner == 'computer':
        game_state["comp_score"] += 1
        game_state["streak"] = 0
        result_msg = "💻 COMPUTER WINS!"
        speak(f"Computer wins with {comp_choice}!")
    else:
        game_state["draws"] += 1
        game_state["streak"] = 0
        result_msg = "🤝 DRAW!"
        speak("It's a draw!")
    
    # Level up logic
    level_msg = ""
    if game_state["user_score"] > 0 and game_state["user_score"] % LEVEL_UP_EVERY == 0:
        old_level = game_state["level"]
        game_state["level"] = (game_state["user_score"] // LEVEL_UP_EVERY) + 1
        if game_state["level"] > old_level:
            level_msg = f"\n\n🔥 LEVEL UP! You're now Level {game_state['level']}! 🔥"
            speak(f"Level {game_state['level']} unlocked!")
    
    # Win streak bonus
    streak_msg = ""
    if game_state["streak"] >= 3:
        streak_msg = f"\n🔥 {game_state['streak']} WIN STREAK! 🔥"
    
    # Build history
    game_state["history"].insert(0, {
        "round": game_state["round"],
        "user": user_choice,
        "comp": comp_choice,
        "result": winner
    })
    
    # Keep only last 10 rounds in display
    if len(game_state["history"]) > 10:
        game_state["history"].pop()
    
    # Format output
    round_summary = f"""
## 🎯 Round {game_state['round']} Result

**You played:** {user_choice} ✊📄✂️  
**Computer played:** {comp_choice} 💻  

### {result_msg}
{level_msg}
{streak_msg}

---
*ML Status:* {'🧠 AI Learning Active' if len(history_user) >= MIN_DATA_FOR_ML else f'📊 Collecting data ({len(history_user)}/{MIN_DATA_FOR_ML})'}
"""
    
    return (
        round_summary,
        IMAGE_URLS[user_choice],
        IMAGE_URLS[comp_choice],
        format_scoreboard(),
        format_history()
    )

def format_scoreboard():
    """Format current game statistics"""
    total_games = game_state["user_score"] + game_state["comp_score"] + game_state["draws"]
    win_rate = (game_state["user_score"] / total_games * 100) if total_games > 0 else 0
    
    return f"""
## 📊 Game Statistics

| Metric | Value |
|--------|-------|
| 🏆 Your Wins | **{game_state['user_score']}** |
| 💻 Computer Wins | {game_state['comp_score']} |
| 🤝 Draws | {game_state['draws']} |
| 🎮 Round | {game_state['round']} |
| ⭐ Level | **{game_state['level']}** |
| 🔥 Win Streak | {game_state['streak']} |
| 📈 Win Rate | {win_rate:.1f}% |
"""

def format_history():
    """Format match history"""
    if not game_state["history"]:
        return "## 📜 Match History\n\nNo games played yet."
    
    history_text = "## 📜 Last 10 Rounds\n\n"
    history_text += "| Round | You | Computer | Result |\n"
    history_text += "|-------|-----|----------|--------|\n"
    
    for entry in game_state["history"]:
        result_emoji = "🏆" if entry["result"] == "user" else ("💻" if entry["result"] == "computer" else "🤝")
        history_text += f"| {entry['round']} | {entry['user']} | {entry['comp']} | {result_emoji} |\n"
    
    return history_text

def reset_game():
    """Reset all game state"""
    game_state.update({
        "user_score": 0,
        "comp_score": 0,
        "draws": 0,
        "round": 0,
        "level": 1,
        "streak": 0,
        "history": []
    })
    history_user.clear()
    history_comp.clear()
    speak("Game reset. Good luck!")
    
    return (
        "## 🔄 Game Reset!\n\nReady for a fresh start. Choose your move!",
        IMAGE_URLS["Blank"],
        IMAGE_URLS["Blank"],
        format_scoreboard(),
        format_history()
    )

def toggle_music():
    """Toggle background music on/off"""
    if not VLC_AVAILABLE or not music_player:
        return "⚠️ VLC not available. Install python-vlc to enable music."
    
    if music_player.is_playing:
        music_player.pause()
        return "⏸️ Music Paused"
    else:
        success = music_player.play()
        if success:
            return "▶️ Music Playing"
        else:
            return "❌ Music failed to start"

def voice_input_handler():
    """Handle voice input and return choice + status"""
    choice, status = listen_for_voice()
    return choice, status

# =====================================
# 🌐 GRADIO INTERFACE
# =====================================
with gr.Blocks(theme=gr.themes.Soft(), title="🎮 Smart ML Voice RPS") as demo:
    gr.Markdown("""
    # 🎮 Smart ML Voice Rock-Paper-Scissors
    ### 🧠 AI-Powered | 🔊 Voice Feedback | 🎵 Music | 📈 Stats & Levels
    
    Play against an AI that **learns from your patterns** and adapts its strategy!
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 🎯 Make Your Move")
            
            # Voice input section
            with gr.Row():
                voice_btn = gr.Button("🎤 SPEAK YOUR MOVE", variant="primary", size="lg")
            
            voice_status = gr.Textbox(
                label="🎤 Voice Status", 
                value="Click 'Speak' to use voice input",
                interactive=False
            )
            
            gr.Markdown("**OR manually select:**")
            
            user_choice = gr.Radio(
                choices=CHOICES,
                label="Click Your Weapon (Auto-plays!)",
                value=None
            )
            
            with gr.Row():
                reset_btn = gr.Button("🔄 Reset Game", variant="secondary")
                music_btn = gr.Button("🎵 Toggle Music", variant="secondary")
            
            music_status = gr.Textbox(label="🎵 Music Status", value="Ready", interactive=False)
            
        with gr.Column(scale=3):
            result_display = gr.Markdown("## 🎯 Choose your move to start!")
    
    with gr.Row():
        user_img = gr.Image(label="👤 Your Move", height=200)
        comp_img = gr.Image(label="💻 Computer's Move", height=200)
    
    with gr.Row():
        scoreboard = gr.Markdown(format_scoreboard())
        match_history = gr.Markdown(format_history())
    
    # Event handlers - Auto-play when user selects choice
    user_choice.change(
        fn=play_round,
        inputs=[user_choice],
        outputs=[result_display, user_img, comp_img, scoreboard, match_history]
    )
    
    # Voice input handler
    voice_btn.click(
        fn=voice_input_handler,
        outputs=[user_choice, voice_status]
    )
    
    reset_btn.click(
        fn=reset_game,
        outputs=[result_display, user_img, comp_img, scoreboard, match_history]
    )
    
    music_btn.click(
        fn=toggle_music,
        outputs=[music_status]
    )
    
    gr.Markdown()

# =====================================
# 🚀 LAUNCH
# =====================================
if __name__ == "__main__":
    # Auto-start music
    if VLC_AVAILABLE and music_player:
        threading.Thread(target=music_player.play, daemon=True).start()
        print("🎵 Background music starting...")
    
    print("🎮 Launching Smart ML Voice RPS Game...")
    print("📍 Opening in your browser...")
    
    try:
        # Try local first
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            inbrowser=True
        )
    except ValueError:
        # If localhost fails, use share mode
        print("🌐 Using public share link (localhost not accessible)...")
        demo.launch(
            share=True,
            show_error=True,
            inbrowser=True
        )
