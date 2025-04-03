import streamlit as st  
import psutil
import time
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="🔥 AI Task Manager", layout="wide", initial_sidebar_state="expanded")

# Custom Styling
st.markdown(
    """
    <style>
        body { background-color: #0e1117; color: white; }
        .stButton>button { background: linear-gradient(45deg, #ff416c, #ff4b2b); color: white; }
        .stDataFrame { border-radius: 10px; overflow: hidden; }
        .stMetric { font-size: 20px; text-align: center; background: #1e2128; border-radius: 10px; padding: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

def collect_training_data():
    process_data = []
    for _ in range(10):
        snapshot = []
        for p in psutil.process_iter(attrs=['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                snapshot.append((p.pid, p.name(), p.cpu_percent(), p.memory_percent()))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue  
        process_data.extend(snapshot)
        time.sleep(0.2)  

    if process_data:
        process_array = np.array(process_data)[:, 2:].astype(float)
        clf = IsolationForest(contamination=0.05, random_state=42)
        clf.fit(process_array)
        joblib.dump(clf, 'anomaly_model.pkl')

def kill_process(pid):
    try:
        p = psutil.Process(pid)
        p.terminate()
        st.success(f"✅ Process {pid} has been terminated.")
    except Exception as e:
        st.error(f"⚠️ Could not terminate process {pid}: {e}")

def monitor_system():
    try:
        clf = joblib.load('anomaly_model.pkl')  
    except FileNotFoundError:
        st.error("❌ Error: Model file not found. Run collect_training_data() first.")
        return None
    
    process_snapshot = []
    for p in psutil.process_iter(attrs=['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            process_snapshot.append((p.pid, p.name(), p.cpu_percent(), p.memory_percent()))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue  

    if process_snapshot:
        process_snapshot.sort(key=lambda x: x[2], reverse=True)  
        process_array = np.array([p[2:] for p in process_snapshot], dtype=float)
        predictions = clf.predict(process_array)
    
        df = pd.DataFrame(process_snapshot, columns=["PID", "Process Name", "CPU Usage", "Memory Usage"])
        df["Status"] = ["✅ Normal" if predictions[i] == 1 else "⚠️ Anomaly" for i in range(len(predictions))]
        return df
    return None

def system_overview():
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    col1, col2 = st.columns(2)
    col1.metric("💻 CPU Usage", f"{cpu_usage}%", delta=f"{'🔥 High' if cpu_usage > 70 else '🟢 Normal'}")
    col2.metric("🗄 Memory Usage", f"{memory.percent}%", delta=f"{'🚨 High' if memory.percent > 80 else '🟢 Normal'}")

def visualize_processes(df):
    if df is not None:
        top_cpu = df.nlargest(5, 'CPU Usage')
        top_mem = df.nlargest(5, 'Memory Usage')
        
        fig1 = px.bar(top_cpu, x='Process Name', y='CPU Usage', title='🔥 Top CPU Consuming Processes', color='CPU Usage')
        fig2 = px.bar(top_mem, x='Process Name', y='Memory Usage', title='🗄 Top Memory Consuming Processes', color='Memory Usage')
        
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig1, use_container_width=True)
        col2.plotly_chart(fig2, use_container_width=True)

        pie_fig = px.pie(df, values='CPU Usage', names='Process Name', title='📊 CPU Usage Distribution')
        st.plotly_chart(pie_fig, use_container_width=True)

def main():
    st.title("🔥 AI Task Manager")
    st.write("### Monitor and manage system processes in real-time with AI-driven anomaly detection.")
    
    system_overview()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🚀 Train AI Model (First Time Only)"):
            with st.spinner("Training model, please wait..."):
                collect_training_data()
                st.success("✅ Model training completed!")
    
    with col2:
        auto_refresh = st.checkbox("🔄 Auto Refresh (every 5s)")
    
    df = monitor_system()
    if df is not None:
        st.dataframe(df.style.applymap(lambda x: 'background-color: #ffcccc' if '⚠️' in str(x) else '', subset=['Status']), height=400, use_container_width=True)
        visualize_processes(df)
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
