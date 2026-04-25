import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from gtts import gTTS
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("🔬 Peel Loss AI Optimization System")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data.csv")
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# =========================
# MODEL
# =========================
X = df.drop("Peel Loss %", axis=1)
y = df["Peel Loss %"]

model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
model.fit(X, y)

# =========================
# INPUT (NO SLIDERS)
# =========================
st.subheader("📝 Enter Process Parameters")

gate = st.number_input("Gate Setting (8–10)", 8, 10, 9)
vsc = st.number_input("VSC % (70–85)", 70, 85, 75)
speed = st.number_input("Roller Speed % (80–100)", 80, 100, 90)
solid = st.number_input("Potato Solid % (18–25)", 18, 25, 20)

# =========================
# PREDICTION
# =========================
inp = pd.DataFrame([{
    "Gate Setting": gate,
    "VSC %": vsc,
    "Roller Speed %": speed,
    "Potato Solid %": solid
}])

pred = model.predict(inp)[0]

st.subheader("🔮 Predicted Peel Loss")

if pred <= 1:
    st.success(f"{pred:.3f}% ✅ Within Target")
else:
    st.error(f"{pred:.3f}% ❌ Above Target")

# =========================
# OPTIMIZATION (≤1%)
# =========================
valid_solutions = []

for sp in range(80, 101, 2):
    for g in range(8, 11):
        for v in range(70, 86, 2):

            temp = pd.DataFrame([{
                "Gate Setting": g,
                "VSC %": v,
                "Roller Speed %": sp,
                "Potato Solid %": solid
            }])

            p = model.predict(temp)[0]

            if p <= 1:
                valid_solutions.append((p, g, v, sp))

# =========================
# BEST SOLUTION
# =========================
if valid_solutions:
    best = sorted(valid_solutions, key=lambda x: x[0])[0]

    st.subheader("⚙️ Optimal Settings")

    st.write(f"Gate: {best[1]}")
    st.write(f"VSC: {best[2]}")
    st.write(f"Speed: {best[3]}")
    st.write(f"Expected Loss: {best[0]:.3f}%")

else:
    best = None
    st.error("❌ No combination achieves ≤1%")

# =========================
# ACTIONS
# =========================
st.subheader("💡 Required Actions")

if best and pred > 1:
    if gate < best[1]:
        st.write("👉 Increase Gate")
    if vsc < best[2]:
        st.write("👉 Increase VSC")
    if speed < best[3]:
        st.write("👉 Increase Speed")
else:
    st.success("Already optimal")

# =========================
# COST
# =========================
if best:
    cost = 0.5*best[3] + 2*best[1] + 100*best[0]
    st.subheader("💰 Cost Score")
    st.write(f"{cost:.2f}")

# =========================
# VISUALS
# =========================
st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("Peel Loss Trend")
    st.line_chart(df["Peel Loss %"])

with col2:
    st.write("Correlation Matrix")
    st.dataframe(df.corr())

# =========================
# HEATMAP
# =========================
st.subheader("🔥 Operating Zone Heatmap")

heat = []

for sp in range(80, 101, 2):
    for v in range(70, 86, 2):

        temp = pd.DataFrame([{
            "Gate Setting": 9,
            "VSC %": v,
            "Roller Speed %": sp,
            "Potato Solid %": solid
        }])

        p = model.predict(temp)[0]
        heat.append([sp, v, p])

heat_df = pd.DataFrame(heat, columns=["Speed","VSC","Loss"])
pivot = heat_df.pivot(index="Speed", columns="VSC", values="Loss")

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r")
st.pyplot(fig)

# =========================
# AI VIDEO + VOICE
# =========================
st.subheader("🎥 AI Assistant")

st.video("assistant.mp4")

def generate_voice(text):
    tts = gTTS(text=text, lang='en', tld='co.in')
    file = "voice.mp3"
    tts.save(file)
    return file

if best:
    message = f"""
    Current peel loss is {pred:.2f} percent.
    Please set gate to {best[1]},
    VSC to {best[2]},
    and speed to {best[3]}.
    """
else:
    message = "No optimal solution found."

st.write("💬", message)

audio = generate_voice(message)
st.audio(audio, autoplay=True)

# =========================
# SAVE DATA (AUTO LEARNING)
# =========================
new_row = pd.DataFrame([{
    "Gate Setting": gate,
    "VSC %": vsc,
    "Roller Speed %": speed,
    "Potato Solid %": solid,
    "Peel Loss %": pred
}])

df = pd.concat([df, new_row], ignore_index=True)
df.to_csv("data.csv", index=False)

# =========================
# COMPARISON
# =========================
st.subheader("📊 Comparison")

col1, col2 = st.columns(2)

with col1:
    st.write("Current")
    st.write(f"{pred:.3f}%")

with col2:
    if best:
        st.write("Optimal")
        st.write(f"{best[0]:.3f}%")
