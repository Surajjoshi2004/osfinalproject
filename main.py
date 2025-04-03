import psutil
import time
import numpy as np
import joblib
import threading
import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

# Function to collect training data
def collect_training_data():
    process_data = []
    for _ in range(15):  # Increased iterations for better training
        snapshot = []
        for p in psutil.process_iter(attrs=['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                snapshot.append((p.pid, p.name(), p.cpu_percent(), p.memory_percent()))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue  # Skip inaccessible processes
        process_data.extend(snapshot)
        time.sleep(0.2)  # Short sleep time for faster training

    if process_data:
        process_array = np.array(process_data)[:, 2:].astype(float)
        clf = IsolationForest(contamination=0.05, random_state=42)
        clf.fit(process_array)
        joblib.dump(clf, 'anomaly_model.pkl')  # Save model

# Function to terminate a process
def kill_process(pid):
    try:
        p = psutil.Process(pid)
        if p.is_running():
            p.terminate()  # Attempt termination
            p.wait(timeout=3)  # Wait for termination
            if p.is_running():
                p.kill()  # Force kill if still running
            st.success(f"✅ Process {pid} has been terminated.")
        else:
            st.warning(f"⚠️ Process {pid} is not running.")
    except psutil.NoSuchProcess:
        st.error(f"❌ Process {pid} does not exist.")
    except psutil.AccessDenied:
        st.error(f"🔒 Access Denied! Run as Administrator.")
    except Exception as e:
        st.error(f"⚠️ Could not terminate process {pid}: {e}")

# Function to monitor processes and detect anomalies
def monitor_system():
    try:
        clf = joblib.load('anomaly_model.pkl')  # Load trained model
    except FileNotFoundError:
        st.error("❌ Model not found! Train the model first.")
        return None

    process_snapshot = []
    for p in psutil.process_iter(attrs=['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            process_snapshot.append((p.pid, p.name(), p.cpu_percent(), p.memory_percent()))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue  

    if process_snapshot:
        process_snapshot.sort(key=lambda x: x[2], reverse=True)  # Sort by CPU usage
        process_array = np.array([p[2:] for p in process_snapshot], dtype=float)
        predictions = clf.predict(process_array)

        df = pd.DataFrame(process_snapshot, columns=["PID", "Process Name", "CPU Usage (%)", "Memory Usage (%)"])
        df["Status"] = ["✅ Normal" if predictions[i] == 1 else "⚠️ Anomaly" for i in range(len(predictions))]
        return df
    return None

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Task Manager", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for styling
    st.markdown("""
        <style>
            .stApp { background-color: #1E1E1E; color: white; }
            .stButton > button { background-color: #ff4b4b; color: white; border-radius: 10px; }
            .stDataFrame { background-color: #2E2E2E; color: white; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🔥 AI Task Manager")
    st.write("### Monitor and manage system processes in real-time with AI-driven anomaly detection.")

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Train AI Model (First Time Only)"):
            with st.spinner("Training model, please wait..."):
                collect_training_data()
                st.success("✅ Model training completed!")

    with col2:
        auto_refresh = st.checkbox("🔄 Auto Refresh (every 5s)")

    df = monitor_system()
    if df is not None:
        st.dataframe(df, height=500, use_container_width=True)
    else:
        st.warning("⚠️ No process data available.")

    process_list = {row[1]: row[0] for row in df.itertuples(index=False)} if df is not None else {}
    selected_process = st.selectbox("🛑 Select a process to kill", options=list(process_list.keys()), index=0 if process_list else None)
    
    if st.button("❌ Kill Selected Process") and selected_process:
        kill_process(process_list[selected_process])

    if auto_refresh:
        time.sleep(5)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
