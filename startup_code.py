import subprocess
import time
import os
import psutil  # Install with: pip install psutil

NGINX_PATH = r"C:/nginx/nginx.exe"  # Adjust path if needed

def is_nginx_running():
    """Check if Nginx is already running."""
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] and 'nginx' in proc.info['name'].lower():
            return True
    return False

def start_nginx():
    """Start Nginx if it's not running."""
    if not is_nginx_running():
        print("Starting Nginx...")
        subprocess.Popen([NGINX_PATH], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print("Nginx is already running.")

if __name__ == "__main__":
    start_nginx()
    
    # Keep monitoring Nginx (optional)
    while True:
        if not is_nginx_running():
            print("Nginx stopped! Restarting...")
            start_nginx()
        time.sleep(10)  # Check every 10 seconds
