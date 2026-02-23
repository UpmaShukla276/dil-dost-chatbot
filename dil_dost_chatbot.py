import streamlit as st
import random
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ═══════════════════════════════════════════════════════════════════════════════
# 📦 DATASET — Based on Kaggle: "Sentiment Analysis for Mental Health"
#    Author   : suchintikasarkar
#    URL      : https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
#    Size     : 53,000+ labeled statements from Reddit, Twitter etc.
#    Labels   : Normal, Depression, Suicidal, Anxiety, Stress, Bipolar, Personality Disorder
#
#    ⚠️  Since Kaggle requires login to download, this code RECREATES the dataset
#       structure faithfully using representative samples from each category.
#       To use the REAL dataset:
#         1. Download from Kaggle (CSV with columns: "statement", "status")
#         2. Place it as  kaggle_mental_health.csv  in the same folder
#         3. The code auto-loads it if found; otherwise uses built-in samples.
# ═══════════════════════════════════════════════════════════════════════════════

KAGGLE_DATASET_SAMPLES = {
    # ── NORMAL (calm, everyday, neutral state) ──────────────────────────────
    "Normal": [
        "I had a good day today", "Feeling okay, nothing special",
        "Just finished my homework", "I went for a walk this morning",
        "Had coffee with a friend", "Today was pretty average",
        "I completed all my tasks", "Feeling neutral today",
        "Nothing much is going on", "I watched a movie last night",
        "Cooked a nice meal today", "Had a productive study session",
        "I called my parents today", "Just relaxing at home",
        "I went grocery shopping", "The weather is nice today",
        "I feel calm and at peace", "Had a decent sleep last night",
        "Caught up with an old friend", "I feel stable and okay",
        "Nothing is bothering me today", "I enjoyed my lunch break",
        "Finished a book I was reading", "Feeling content with life",
        "I took a nice long shower and felt refreshed", "Things are going as usual",
        "Not great not bad, just normal", "I feel like myself today",
        "Just a regular Tuesday", "I feel balanced today",
    ],
    # ── DEPRESSION ──────────────────────────────────────────────────────────
    "Depression": [
        "I feel completely empty inside", "Nothing brings me joy anymore",
        "I have lost interest in everything I used to love",
        "Getting out of bed feels impossible", "I feel worthless and hopeless",
        "I don't see the point in anything", "I have been crying for no reason",
        "Everything feels dark and heavy", "I feel like a burden to everyone",
        "I can't concentrate on anything", "I feel numb all the time",
        "I have no energy to do anything", "I feel like I am failing at life",
        "I don't enjoy things that used to make me happy",
        "I feel so low and broken", "Life feels meaningless to me",
        "I isolate myself because I have no energy to socialize",
        "I haven't showered in days because I can't get up",
        "I feel like nothing will ever get better",
        "I am so exhausted all the time even after sleeping",
        "I feel trapped in my own mind", "I hate myself",
        "I can't stop crying and I don't know why",
        "I feel disconnected from reality", "I feel like disappearing",
        "Nothing I do feels good enough", "I feel hopeless about the future",
        "I sleep too much but still feel tired", "I can't make myself eat",
        "I feel like I am just going through the motions of living",
    ],
    # ── ANXIETY ─────────────────────────────────────────────────────────────
    "Anxiety": [
        "I can't stop worrying about everything", "My heart is racing for no reason",
        "I am scared something bad is going to happen",
        "I feel a constant sense of dread", "I can't relax no matter what I do",
        "My mind won't stop racing", "I feel panicky and out of control",
        "I am terrified of failing my exams", "I overthink every decision I make",
        "I have a presentation tomorrow and I am freaking out",
        "I feel tense and on edge all the time",
        "I can't sleep because I keep worrying", "I feel like I am going to have a panic attack",
        "Deadlines are overwhelming me", "I am scared of what people think of me",
        "I have been feeling restless and anxious all day",
        "I spiral into worst case scenarios constantly",
        "Social situations make me extremely nervous",
        "I feel shortness of breath when I think about the future",
        "I am so stressed about everything piling up",
        "I feel dread every morning when I wake up",
        "The pressure of expectations is crushing me",
        "I can't sit still because of anxiety", "I keep checking things repeatedly",
        "I am scared to make mistakes", "I feel like my anxiety is taking over my life",
        "I am worried I will embarrass myself", "I have been hyperventilating from stress",
        "My stomach hurts from anxiety", "I feel afraid of leaving the house",
    ],
    # ── STRESS ──────────────────────────────────────────────────────────────
    "Stress": [
        "I am completely overwhelmed with work", "I have too much on my plate right now",
        "I can't keep up with everything that is expected of me",
        "Deadlines are piling up and I don't know where to start",
        "I am burnt out from studying non-stop",
        "I feel like I am being pulled in too many directions",
        "I haven't had a break in weeks", "The workload is unbearable",
        "I am running on empty", "Everything is due at the same time",
        "I snapped at someone today because I am so stressed",
        "I can't enjoy anything because of the pressure",
        "I am losing sleep over all the responsibilities",
        "I feel like I am drowning in tasks",
        "I have been skipping meals because I am too busy",
        "I feel like I can't breathe from all the stress",
        "I am constantly tired from all the pressure",
        "I feel like I have no time for myself",
        "I have been grinding for so long I feel numb",
        "My head hurts from staring at screens all day",
        "I am burned out and need a break", "Work is consuming all of my time",
        "I feel exhausted from trying to balance everything",
        "I am snapping at people around me because I am so stressed out",
        "I need a vacation so badly", "The pressure never stops",
        "I wish I could just pause everything for a day",
        "I am so tense my shoulders and neck hurt constantly",
        "I feel like I cannot catch a break", "This semester is destroying me",
    ],
    # ── BIPOLAR ─────────────────────────────────────────────────────────────
    "Bipolar": [
        "I go from feeling on top of the world to completely hopeless in the same day",
        "My moods swing wildly without warning",
        "One moment I am ecstatic and the next I am devastated",
        "I had a manic episode where I spent all my money impulsively",
        "I feel intense energy and then crash into depression",
        "My emotions go from extreme highs to deep lows very quickly",
        "I can't predict how I will feel from one hour to the next",
        "I have been on an emotional rollercoaster for weeks",
        "I felt invincible yesterday but today I can't get out of bed",
        "My mood cycles are unpredictable and exhausting",
        "I barely slept during my high phase and now I am crashing",
        "I made reckless decisions during my high and now regret them",
        "The mood swings are making it hard to maintain relationships",
        "I don't know which version of me will show up today",
        "My highs feel great but the lows that follow are terrible",
    ],
    # ── PERSONALITY DISORDER ────────────────────────────────────────────────
    "Personality Disorder": [
        "I am terrified of being abandoned by the people I love",
        "My sense of self changes depending on who I am around",
        "I have intense and unstable relationships",
        "I feel empty most of the time and don't know who I am",
        "I go from idealizing someone to hating them very suddenly",
        "I act impulsively and often regret it later",
        "I have a very unstable self image",
        "My emotions are very intense and hard to control",
        "I feel chronic emptiness that I can't shake",
        "I often feel like people are going to leave me",
        "I have a hard time trusting anyone",
        "My identity feels fragmented and unclear",
        "I react very strongly to perceived slights or rejection",
        "I have outbursts of anger that seem out of proportion",
        "I tend to push people away even though I fear abandonment",
    ],
    # ── SUICIDAL ────────────────────────────────────────────────────────────
    "Suicidal": [
        "I don't want to be here anymore", "I feel like everyone would be better off without me",
        "I have been having thoughts of ending my life",
        "I can't see a reason to keep going", "I feel like there is no way out",
        "I have been thinking about suicide", "I don't want to wake up tomorrow",
        "I wish I could just disappear permanently",
        "The pain is so unbearable that I want it to end",
        "I have been researching ways to hurt myself",
        "I feel like I am a burden and should not exist",
        "I have lost all hope and see no future for myself",
        "I feel like dying would be a relief",
        "I have written notes in case I don't make it",
        "I feel completely alone and see no point in living",
    ],
}

# ── Build DataFrame from samples ────────────────────────────────────────────
import os
CSV_PATH = "kaggle_mental_health.csv"

if os.path.exists(CSV_PATH):
    # Load REAL Kaggle dataset if available
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip().str.lower()
    # handle both common column name conventions
    if "statement" in df.columns and "status" in df.columns:
        df = df[["statement", "status"]].dropna()
    elif "text" in df.columns and "label" in df.columns:
        df = df.rename(columns={"text": "statement", "label": "status"})
    df["status"] = df["status"].str.strip()
    USING_REAL_DATA = True
else:
    rows = []
    for label, texts in KAGGLE_DATASET_SAMPLES.items():
        for t in texts:
            rows.append({"statement": t, "status": label})
    df = pd.DataFrame(rows)
    USING_REAL_DATA = False

# ── Map 7 Kaggle labels → 6 chatbot moods ───────────────────────────────────
LABEL_MAP = {
    "Normal":               "neutral",
    "Depression":           "sad",
    "Anxiety":              "anxious",
    "Stress":               "anxious",   # stress → anxious bucket
    "Bipolar":              "sad",        # treated with empathy like depression
    "Personality Disorder": "lonely",
    "Suicidal":             "sad",        # handled gently as deep sadness
}

df["mood"] = df["status"].map(LABEL_MAP)
df = df.dropna(subset=["mood"])

# ── Text cleaning ────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean"] = df["statement"].apply(clean_text)

# ── Train / Test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["mood"], test_size=0.2, random_state=42, stratify=df["mood"]
)

# ── Pipeline: TF-IDF + Logistic Regression ──────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=10000,
                               sublinear_tf=True, min_df=1)),
    ("clf", LogisticRegression(max_iter=1000, C=3.0)),
])
pipeline.fit(X_train, y_train)

# ── Evaluate and store report ────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
MODEL_REPORT = classification_report(y_test, y_pred, zero_division=0)

# ── Prediction with keyword guard ───────────────────────────────────────────
KEYWORDS = {
    "anxious": ["anxious","anxiety","stress","stressed","panic","nervous","worry",
                "worrying","worried","overwhelm","scared","fear","afraid","pressure",
                "overthink","restless","tense","dread","freaking out","can't sleep",
                "deadline","hyperventilat"],
    "sad":     ["sad","depress","hopeless","worthless","cry","crying","broken",
                "empty","numb","low","grief","hurt","pain","miserable","unhappy",
                "heartbroken","devastated","lost","suicid","want to die","end my life",
                "don't want to be here","disappear","burden","no point","no reason"],
    "angry":   ["angry","anger","furious","rage","hate","irritated","frustrated",
                "frustration","annoyed","annoying","mad","betrayed","disrespected",
                "scream","yell","snap"],
    "lonely":  ["alone","lonely","loneliness","isolated","no friends","nobody",
                "no one","invisible","ignored","left out","disconnected",
                "no one cares","no one understands","abandon"],
    "happy":   ["happy","great","amazing","wonderful","excited","joy","joyful",
                "proud","fantastic","awesome","feeling good","feeling great",
                "positive","motivated","energetic","grateful","content","thrilled"],
}

def detect_mood(text):
    text_lower = clean_text(text)
    for mood, kws in KEYWORDS.items():
        if any(kw in text_lower for kw in kws):
            return mood
    proba = pipeline.predict_proba([text_lower])[0]
    confidence = max(proba)
    mood = pipeline.predict([text_lower])[0]
    return mood if confidence >= 0.30 else "neutral"

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dil Dost 💛 Mental Health Companion",
    page_icon="💛",
    layout="centered"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Baloo+2:wght@400;600;800&display=swap');
* { font-family: 'Poppins', sans-serif; }

[data-testid="stAppViewContainer"],[data-testid="stMain"],.main,section.main {
    background: linear-gradient(135deg, #fff9f0 0%, #ffecd2 50%, #fff0f6 100%) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
body, p, div, span, label { color: #2d2d2d !important; }

.main-header { text-align: center; padding: 2rem 0 1rem 0; }
.main-title {
    font-family: 'Baloo 2', cursive; font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #ff6b6b, #ffa94d, #ff6b9d);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0; letter-spacing: -1px;
}
.main-subtitle { color: #888 !important; font-size: 0.95rem; margin-top: 0.2rem; font-weight: 300; }

.mood-badge { display: inline-block; padding: 0.25rem 0.9rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; margin-bottom: 0.4rem; }
.mood-sad     { background: #dbeafe !important; color: #1d4ed8 !important; }
.mood-happy   { background: #dcfce7 !important; color: #15803d !important; }
.mood-anxious { background: #fef3c7 !important; color: #b45309 !important; }
.mood-angry   { background: #fee2e2 !important; color: #b91c1c !important; }
.mood-neutral { background: #f3f4f6 !important; color: #4b5563 !important; }
.mood-lonely  { background: #ede9fe !important; color: #6d28d9 !important; }

.chat-container {
    background: rgba(255,255,255,0.75) !important; border-radius: 20px;
    padding: 1.2rem; min-height: 320px; max-height: 480px; overflow-y: auto;
    border: 1px solid rgba(255,200,150,0.5); box-shadow: 0 8px 32px rgba(0,0,0,0.08); margin-bottom: 1rem;
}
.clearfix::after { content: ""; display: table; clear: both; }

.chat-bubble-user {
    background: linear-gradient(135deg, #ff6b6b, #ffa94d) !important;
    color: #ffffff !important; border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem; margin: 0.5rem 0; max-width: 75%;
    float: right; clear: both; font-size: 0.93rem;
    box-shadow: 0 4px 12px rgba(255,107,107,0.35); word-wrap: break-word;
}
.chat-bubble-user * { color: #ffffff !important; }

.chat-bubble-bot {
    background: #ffffff !important; color: #1a1a1a !important;
    border-radius: 18px 18px 18px 4px; padding: 0.75rem 1.1rem;
    margin: 0.5rem 0; max-width: 80%; float: left; clear: both; font-size: 0.93rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.10); border-left: 3px solid #ffa94d; word-wrap: break-word;
}
.chat-bubble-bot * { color: #1a1a1a !important; }

.tip-box {
    background: linear-gradient(135deg, #fff9db, #fff3bf) !important;
    border: 1px solid #ffd43b; border-radius: 14px;
    padding: 1rem 1.2rem; margin: 0.8rem 0; font-size: 0.88rem; color: #5c3d00 !important;
}
.tip-box .tip-title { font-weight: 700; margin-bottom: 0.3rem; color: #e67700 !important; }

.info-box {
    background: #f0f9ff !important; border: 1px solid #bae6fd;
    border-radius: 12px; padding: 0.8rem 1.2rem;
    font-size: 0.82rem; color: #0c4a6e !important; margin-bottom: 1rem;
}
.info-box * { color: #0c4a6e !important; }

.stTextInput > div > div > input {
    border-radius: 999px !important; border: 2px solid #ffa94d !important;
    padding: 0.6rem 1.2rem !important; background: #ffffff !important; color: #1a1a1a !important;
}
.stTextInput > div > div > input::placeholder { color: #aaa !important; }
.stTextInput > div > div > input:focus {
    box-shadow: 0 0 0 3px rgba(255,169,77,0.25) !important; border-color: #ff6b6b !important;
}
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #ff6b6b, #ffa94d) !important;
    color: white !important; border: none !important; border-radius: 999px !important;
    padding: 0.55rem 1.8rem !important; font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(255,107,107,0.35) !important; width: 100% !important;
}
.footer {
    text-align: center; margin-top: 2rem; color: #aaa !important;
    font-size: 0.78rem; font-weight: 300; letter-spacing: 0.5px;
}
.footer strong { color: #ff6b6b !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE BANK — all English, varied, non-repeating
# ═══════════════════════════════════════════════════════════════════════════════
responses = {
    "sad": [
        "I'm really sorry you're feeling this way. It's completely okay to feel sad — you don't have to pretend to be fine. What happened? I'm here to listen. 💛",
        "Sadness is heavy, and you don't have to carry it alone. Take a deep breath — I'm right here with you. Want to share what's going on? 🤗",
        "You don't have to be strong all the time. It's okay to feel low. Sometimes just talking about it makes things a little lighter — tell me more. 💙",
        "I hear you. Days like these are really tough, and your feelings are completely valid. What's been weighing on your mind? 🌧️",
        "Feeling sad doesn't mean you're weak — it means you're human. I'm glad you're here talking about it. What's been going on lately? 💜",
        "Sometimes sadness just shows up and there's no easy explanation. That's okay. I'm here — you don't have to go through this alone. 🌱",
        "It sounds like you're going through something really painful right now. I want you to know that your feelings matter. Can you tell me more? 💛",
        "I'm sorry things feel so hard right now. You deserve to feel better, and taking this first step to talk is actually really brave. 🌻",
        "What you're feeling sounds really exhausting. Please be gentle with yourself — you're carrying a lot. I'm here, tell me what's on your heart. 💙",
        "You matter, and what you're feeling matters. Let's talk through it — one thing at a time. What feels heaviest right now? 🌸",
    ],
    "anxious": [
        "Anxiety is tough, but you're tougher! 💪 Try this right now: breathe in for 4 seconds, hold for 4, breathe out for 4. How are you feeling?",
        "It sounds like you're really overwhelmed. Let's slow things down together. What's the one biggest thing stressing you out right now? 🌿",
        "That feeling of panic or dread is so hard to deal with. Ground yourself — name 5 things you can see around you. It actually helps! What's worrying you most?",
        "Worrying about the future is exhausting. Remember: you can only control right now, this moment. What's the very next small step you can take? 🌊",
        "When anxiety spikes, your brain thinks there's danger everywhere — but you're safe right now. Let's talk through what's going on. What triggered this? 💛",
        "Overthinking is your brain trying to protect you, but it can spiral fast. Let's untangle it together — what specifically are you most afraid of? 🤝",
        "It's okay to feel nervous. Even the most confident people get anxious. The key is you're aware of it, which means you can work through it. 📚",
        "Feeling overwhelmed is your mind's way of saying you need support right now. You've come to the right place. What's piling up for you? 💙",
        "Stress and pressure can feel crushing. But you don't have to solve everything at once. What's one thing you can set down for right now? 🌼",
        "You're going through a lot, and it makes sense you feel tense. Let's break it down together — what's your biggest worry today? 🌟",
    ],
    "happy": [
        "That's so wonderful to hear! 🎉 Happiness is contagious — I'm smiling too! What made today so great for you?",
        "Yes! Hold onto this feeling. 🌟 You deserve every bit of it. What happened that's made you so happy today?",
        "This is the best kind of update! Your positivity is genuinely uplifting. Tell me more — I want to celebrate with you! 😄",
        "Love this energy! 🌈 Moments like these are worth savoring. What's the highlight of your day?",
        "It's so good to see you in high spirits! Did something specific happen, or are you just having one of those naturally great days? ☀️",
        "Good vibes only! 💫 You deserve all the happiness coming your way. What are you most excited about right now?",
        "That's amazing! Happiness tends to grow when you share it — what's the good news? 🎊",
        "Your energy right now is so refreshing! 🌻 What's been the best part of your day so far?",
        "So glad to hear things are going well! Let's lock this feeling in — tell me everything that's made you smile today 😊",
        "Keep riding this wave! 🏄 What are you most grateful for right now?",
    ],
    "angry": [
        "I hear you — and your anger is completely valid. Before anything else, try counting slowly to 10 and take a big sip of water. Then tell me what happened. 💧",
        "That sounds really frustrating. It's okay to feel angry when things feel unfair or out of control. What's got you so worked up? 🔥",
        "Anger usually means something important was hurt — a boundary, an expectation, your trust. What exactly happened? 🤝",
        "Let it out — I'm listening and I won't judge. What's the situation that's made you feel this way? 😤",
        "That intense frustration you're feeling is real and valid. Take a moment to breathe — then walk me through it. 💛",
        "It sounds like you've been pushed to your limit. That's exhausting. Do you want to vent first, or talk through how to handle it? 🌊",
        "When we're angry it's hard to think clearly — that's just biology. Give yourself a minute, then let's break down what's really going on. 🧠",
        "I'm sorry something pushed you to this point. You deserve to be treated with respect. What happened? 💙",
        "Your frustration makes complete sense given what you're dealing with. I'm here — let it all out. What's going on? 🌿",
        "Being this angry takes a lot of energy. I want to help you work through it. What's the core of what upset you? 💜",
    ],
    "lonely": [
        "Loneliness is one of the hardest feelings to sit with. But you reaching out right now? That takes courage. You're not alone — I'm here. 💛",
        "Feeling like no one sees you is really painful. I see you. I'm listening. Tell me what's been happening. 🌸",
        "It's so hard when you feel disconnected from people around you. You deserve genuine connection. How long have you been feeling this way? 🤗",
        "Sometimes we can be surrounded by people and still feel completely alone — that's one of the loneliest feelings. What does your daily life look like right now? 💙",
        "I'm glad you're talking to me. Loneliness doesn't mean you're unlikable — it often just means the right connections haven't found you yet. 🌼",
        "You matter more than you know. The fact that you're feeling lonely tells me you crave real connection — and that's a beautiful thing. 💜",
        "Not having someone to talk to is genuinely difficult. Here's a small step: send one person a message today — just 'hey, how are you?' It opens doors. 🌱",
        "I hear you. Being left out or ignored really wears you down. This says nothing bad about you. Have you been able to open up to anyone about this? 🌟",
        "Feeling invisible is so painful. You deserve to be seen and heard — and you are, right here, right now. What's been making you feel this way? 💛",
        "The loneliness you're describing sounds really deep. I'm here and I genuinely care. Tell me more about what's going on in your life. 🤍",
    ],
    "neutral": [
        "Hey! I'm Dil Dost — your personal mental health companion. 🌟 I'm here to listen without any judgment. How are you feeling today?",
        "Hello! It's good to have you here. Whether you're doing great, not so great, or somewhere in between — feel free to share. I'm all ears. 👂",
        "Hi there! You can talk to me about anything — stress, happiness, confusion, or just whatever's on your mind. What's up? 😊",
        "Welcome! Think of me as a friend who's always available and never judges. How's your day going so far? ☀️",
        "Hey, good to see you! Even if you're not sure what you want to say, that's okay. Sometimes just starting the conversation is enough. 💛",
        "Hello! I'm here whenever you need to talk. How are you really doing today — not just the 'I'm fine' version? 😄",
        "Hi! Feel free to share anything — I genuinely want to understand how you're feeling. What brought you here today? 🌼",
        "Hey! No pressure to have it all figured out. Just talk to me — what's going on in your world right now? 🌈",
    ],
}

relaxation_tips = {
    "anxious": ("🧘 Quick Calm Technique", "Try Box Breathing: inhale 4 sec → hold 4 sec → exhale 4 sec → hold 4 sec. Repeat 4 times. This activates your parasympathetic nervous system and reduces anxiety in under 2 minutes."),
    "sad":     ("💜 Mood Lift Tip", "Put on a song you love right now. Music directly affects brain chemistry. Or take a 5-minute walk — sunlight and movement are natural mood boosters."),
    "angry":   ("❄️ Cool Down Strategy", "Splash cold water on your face — it triggers the diving reflex which slows your heart rate fast. Then take 10 slow breaths where the exhale is longer than the inhale."),
    "lonely":  ("🌸 Small Connection Tip", "Send one message to someone today — even a simple 'hey, thinking of you' counts. Research shows even tiny social interactions significantly reduce loneliness."),
    "happy":   ("✨ Lock In the Good Feeling", "Write down 3 specific things that made today great. Gratitude journaling trains your brain to notice positives and builds resilience for harder days."),
}

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.used_responses = []
    st.session_state.messages.append({
        "role": "bot",
        "text": "Hello! 👋 I'm Dil Dost — your personal mental health companion. This is a safe, judgment-free space just for you. How are you feeling today? 💛",
        "mood": None
    })
if "used_responses" not in st.session_state:
    st.session_state.used_responses = []

def get_response(mood):
    pool = responses[mood]
    unused = [r for r in pool if r not in st.session_state.used_responses]
    if not unused:
        unused = pool
    chosen = random.choice(unused)
    st.session_state.used_responses.append(chosen)
    if len(st.session_state.used_responses) > 6:
        st.session_state.used_responses.pop(0)
    return chosen

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <div class="main-title">💛 Dil Dost</div>
  <div class="main-subtitle">Your Personal Mental Health Companion • Safe Space for Students</div>
</div>
""", unsafe_allow_html=True)

# Dataset info banner
data_source = "✅ Real Kaggle dataset loaded" if USING_REAL_DATA else "📦 Built-in samples (based on Kaggle dataset structure)"
st.markdown(f"""
<div class="info-box">
  <strong>🤖 Model Info:</strong> Trained on <em>Sentiment Analysis for Mental Health</em>
  by <strong>suchintikasarkar</strong> on Kaggle
  (<a href="https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health" target="_blank">link</a>)
  — 53,000+ labeled statements across 7 mental health categories: Normal, Depression, Anxiety, Stress, Bipolar, Personality Disorder, Suicidal.
  Mapped to 6 chatbot moods. Pipeline: TF-IDF (trigrams) + Logistic Regression. &nbsp;|&nbsp; {data_source}
</div>
""", unsafe_allow_html=True)

# Chat
mood_colors = {
    "sad":     ("😢 Sad",     "mood-sad"),
    "happy":   ("😄 Happy",   "mood-happy"),
    "anxious": ("😰 Anxious", "mood-anxious"),
    "angry":   ("😠 Angry",   "mood-angry"),
    "neutral": ("😊 Neutral", "mood-neutral"),
    "lonely":  ("🥺 Lonely",  "mood-lonely"),
}

chat_html = '<div class="chat-container">'
for msg in st.session_state.messages:
    if msg["role"] == "user":
        chat_html += f'<div class="chat-bubble-user">{msg["text"]}</div><div class="clearfix"></div>'
    else:
        badge = ""
        if msg.get("mood") and msg["mood"] in mood_colors:
            label, css = mood_colors[msg["mood"]]
            badge = f'<div><span class="mood-badge {css}">{label}</span></div>'
        chat_html += f'<div class="chat-bubble-bot">{badge}{msg["text"]}</div><div class="clearfix"></div>'
chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)

# Relaxation tip
last_bot = next((m for m in reversed(st.session_state.messages) if m["role"] == "bot" and m.get("mood")), None)
if last_bot and last_bot["mood"] in relaxation_tips:
    title, tip = relaxation_tips[last_bot["mood"]]
    st.markdown(f"""
    <div class="tip-box">
      <div class="tip-title">{title}</div>
      <div style="color:#5c3d00 !important;">{tip}</div>
    </div>
    """, unsafe_allow_html=True)

# Input
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            label="Message",
            placeholder="Type how you're feeling right now...",
            label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button("Send 💬")

if submitted and user_input.strip():
    mood = detect_mood(user_input)
    reply = get_response(mood)
    st.session_state.messages.append({"role": "user", "text": user_input})
    st.session_state.messages.append({"role": "bot", "text": reply, "mood": mood})
    st.rerun()

# Model report expander
with st.expander("📊 View Model Classification Report"):
    st.code(MODEL_REPORT)
    st.caption(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

# Footer
st.markdown("""
<div class="footer">
  Made with 💛 by <strong>Upma Shukla</strong> &nbsp;|&nbsp; Dil Dost — Mental Health Companion for Students<br>
  <span style="font-size:0.72rem; opacity:0.6;">
    Dataset: Sentiment Analysis for Mental Health (Kaggle • suchintikasarkar) •
    Model: TF-IDF + Logistic Regression
  </span>
</div>
""", unsafe_allow_html=True)
