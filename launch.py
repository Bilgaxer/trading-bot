import subprocess
import sys
import time
import socket
import os

def get_ip():
    # Get local IP address
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def start_bot():
    print("Starting trading bot...")
    bot_process = subprocess.Popen([sys.executable, "bot.py"])
    return bot_process

def start_dashboard(port=8501):
    print("Starting dashboard...")
    # Configure Streamlit to be accessible from network
    dashboard_process = subprocess.Popen([
        sys.executable, 
        "-m", 
        "streamlit", 
        "run", 
        "dashboard.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0"
    ])
    return dashboard_process

def main():
    port = 8501
    local_ip = get_ip()
    
    # Start bot and dashboard
    bot_process = start_bot()
    time.sleep(2)  # Wait for bot to initialize
    dashboard_process = start_dashboard(port)
    
    print("\n✅ Trading bot and dashboard are running!")
    print(f"\n🌐 Dashboard accessible at: http://{local_ip}:{port}")
    print("📱 To access from your phone:")
    print(f"1. Make sure your phone is on the same WiFi network as this computer")
    print(f"2. Open your phone's browser and go to: http://{local_ip}:{port}")
    print("\nPress Ctrl+C to stop all processes...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        bot_process.terminate()
        dashboard_process.terminate()

if __name__ == "__main__":
    main() 