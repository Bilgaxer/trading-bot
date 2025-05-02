import subprocess
import sys
import time
import socket
import os
import argparse
import psutil  # Add psutil for process management

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

def kill_existing_streamlit():
    # Kill any existing Streamlit processes to avoid port conflicts
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and 'streamlit' in ' '.join(cmdline):
                print(f"Terminating existing Streamlit process (PID: {proc.info['pid']})")
                psutil.Process(proc.info['pid']).terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def start_bot():
    print("Starting trading bot...")
    bot_process = subprocess.Popen([sys.executable, "bot.py"])
    return bot_process

def start_dashboard(dashboard_type="local", port=8501):
    dashboard_file = "dashboard.py" if dashboard_type == "local" else "cloud_dashboard.py"
    print(f"Starting {dashboard_type} dashboard on port {port}...")
    
    python_exe = sys.executable
    dashboard_process = subprocess.Popen([
        python_exe,
        "-m", "streamlit",
        "run",
        dashboard_file,
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--server.headless", "true",
        "--browser.serverAddress", get_ip(),
        "--browser.serverPort", str(port),
        "--browser.gatherUsageStats", "false"
    ])
    return dashboard_process

def create_streamlit_config(port, local_ip):
    # Create .streamlit config directory if it doesn't exist
    os.makedirs(".streamlit", exist_ok=True)
    
    # Create/update config.toml for network access
    config_path = os.path.join(".streamlit", "config.toml")
    with open(config_path, "w") as f:
        f.write(f"""
[server]
enableCORS = false
enableXsrfProtection = false
address = "0.0.0.0"
port = {port}
headless = true
baseUrlPath = ""
maxUploadSize = 200
maxMessageSize = 200

[browser]
serverAddress = "{local_ip}"
serverPort = {port}
gatherUsageStats = false
        """)
    print(f"Created Streamlit config at: {os.path.abspath(config_path)}")

def check_streamlit_installation():
    try:
        # Check if streamlit is installed
        result = subprocess.run(
            [sys.executable, "-c", "import streamlit; print('Streamlit found')"],
            capture_output=True,
            text=True
        )
        if "Streamlit found" in result.stdout:
            print("✅ Streamlit installation verified")
            return True
        else:
            print("❌ Streamlit not found. Please install with: pip install streamlit")
            return False
    except Exception as e:
        print(f"❌ Error checking Streamlit installation: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Launch trading bot and dashboard')
    parser.add_argument('--dashboard', choices=['local', 'cloud'], default='local',
                        help='Dashboard type to run (default: local)')
    parser.add_argument('--port', type=int, default=8501,
                        help='Port for Streamlit dashboard (default: 8501)')
    parser.add_argument('--bot-only', action='store_true',
                        help='Run only the trading bot without dashboard')
    parser.add_argument('--dashboard-only', action='store_true',
                        help='Run only the dashboard without trading bot')
                        
    args = parser.parse_args()
    port = args.port
    local_ip = get_ip()
    
    print(f"\n🖥️  Your local IP address is: {local_ip}")
    
    # Kill any existing Streamlit processes
    if not args.bot_only:
        kill_existing_streamlit()
    
    # Create/update Streamlit config
    create_streamlit_config(port, local_ip)
    
    # Verify Streamlit installation
    if not args.bot_only and not check_streamlit_installation():
        print("❌ Cannot proceed with dashboard launch due to Streamlit installation issues")
        if not args.dashboard_only:
            print("⚠️ Starting bot only...")
            bot_process = start_bot()
            print("✅ Trading bot started successfully")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n⏹️  Shutting down...")
                bot_process.terminate()
                print("✅ Bot stopped")
                print("👋 Goodbye!")
            return
        else:
            return
    
    # Start bot and dashboard based on arguments
    bot_process = None
    dashboard_process = None
    
    if not args.dashboard_only:
        try:
            bot_process = start_bot()
            time.sleep(2)  # Wait for bot to initialize
            print("✅ Trading bot started successfully")
        except Exception as e:
            print(f"❌ Error starting trading bot: {e}")
        
    if not args.bot_only:
        try:
            dashboard_process = start_dashboard(args.dashboard, port)
            print(f"\n✅ Dashboard ({args.dashboard.upper()} mode) started")
            print(f"\n🌐 Access the dashboard at:")
            print(f"   • Local machine: http://localhost:{port}")
            print(f"   • Other devices on your network: http://{local_ip}:{port}")
            print("\n📱 To access from your phone:")
            print(f"  1. Make sure your phone is on the same WiFi network as this computer")
            print(f"  2. Open your phone's browser and go to: http://{local_ip}:{port}")
            print(f"  3. If that doesn't work, try: http://localhost:{port}")
            
            # Check if dashboard is actually accessible
            print("\n🔍 Verifying dashboard accessibility...")
            time.sleep(5)  # Give Streamlit time to start
            try:
                import requests
                response = requests.get(f"http://{local_ip}:{port}", timeout=5)
                if response.status_code == 200:
                    print("✅ Dashboard is accessible!")
                else:
                    print(f"⚠️ Dashboard responded with status code: {response.status_code}")
                    print("   This may indicate configuration issues.")
            except Exception as e:
                print(f"⚠️ Could not verify dashboard access: {e}")
                print("   This doesn't necessarily mean the dashboard isn't working.")
                print("   Try accessing it manually using the URLs above.")
        except Exception as e:
            print(f"❌ Error starting dashboard: {e}")
    
    print("\n🔥 Services started. Press Ctrl+C to stop all processes...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
        if bot_process:
            bot_process.terminate()
            print("✅ Bot stopped")
        if dashboard_process:
            dashboard_process.terminate()
            print("✅ Dashboard stopped")
        print("👋 Goodbye!")

if __name__ == "__main__":
    main() 