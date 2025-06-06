# core/john_wrapper.py
import subprocess
import os
import logging
import multiprocessing as mp
import shutil
from utils.logger import setup_logging

logger = setup_logging("output/logs/passripperhydra.log")

def crack_hashes(hash_file, passwords, num_processes=4, use_gpu=True):
    """
    Crack hashes using John the Ripper (CPU) or Hashcat (GPU) with parallel processing.
    Args:
        hash_file (str): Path to the hash file.
        passwords (list): List of passwords to use as wordlist.
        num_processes (int): Number of parallel processes (for CPU mode).
        use_gpu (bool): Use GPU acceleration with Hashcat if available.
    Returns:
        list: Cracked results.
    """
    results = []
    pwd_file = "temp_passwords.txt"
    try:
        logger.info(f"Starting offline cracking for {hash_file} (GPU: {use_gpu})...")
        with open(pwd_file, "w") as f:
            for pwd in passwords:
                f.write(f"{pwd}\n")

        # Check if Hashcat is available and GPU mode is enabled
        hashcat_available = shutil.which("hashcat") is not None
        if use_gpu and hashcat_available:
            logger.info("Using Hashcat for GPU-accelerated cracking...")
            # Hashcat hash modes (equivalent to John's formats)
            hash_modes = {
                "raw-md5": "0",
                "raw-sha1": "100",
                "raw-sha256": "1400",
                "raw-sha512": "1700",
                "nt": "1000",
                "sha256crypt": "7400",
                "bcrypt": "3200"
            }
            # Try each hash mode
            for hash_type, mode in hash_modes.items():
                try:
                    cmd = [
                        "hashcat", "-m", mode, "-a", "0", "--potfile-disable",
                        hash_file, pwd_file, "--quiet"
                    ]
                    subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
                    # Extract results from Hashcat output
                    cmd_show = ["hashcat", "-m", mode, "--show", hash_file]
                    output = subprocess.run(cmd_show, capture_output=True, text=True, check=True).stdout
                    for line in output.splitlines():
                        if ":" in line:
                            parts = line.split(":")
                            username = parts[0] if len(parts) > 2 else "N/A"
                            password = parts[-1]
                            results.append({
                                "target": hash_file,
                                "mode": "Offline (GPU)",
                                "username": username,
                                "password": password,
                                "status": "Cracked"
                            })
                    break
                except subprocess.SubprocessError:
                    continue
        else:
            logger.info("Using John the Ripper for CPU-based cracking...")
            # Split hash file into chunks for parallel processing
            with open(hash_file, "r") as f:
                hashes = f.readlines()
            chunk_size = len(hashes) // num_processes
            hash_chunks = [hashes[i:i + chunk_size] for i in range(0, len(hashes), chunk_size)]

            def process_chunk(chunk, chunk_id):
                chunk_file = f"temp_hashes_{chunk_id}.txt"
                with open(chunk_file, "w") as f:
                    f.writelines(chunk)
                chunk_results = []
                hash_formats = ["raw-md5", "raw-sha1", "raw-sha256", "raw-sha512", "nt", "sha256crypt", "bcrypt"]
                for fmt in hash_formats:
                    try:
                        cmd = ["john", f"--wordlist={pwd_file}", f"--format={fmt}", chunk_file]
                        subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
                        cmd_show = ["john", "--show", chunk_file]
                        output = subprocess.run(cmd_show, capture_output=True, text=True, check=True).stdout
                        for line in output.splitlines():
                            if ":" in line and not line.startswith("#"):
                                username, password = line.split(":", 1)[0], line.split(":", 1)[1].split(":")[0]
                                chunk_results.append({
                                    "target": hash_file,
                                    "mode": "Offline (CPU)",
                                    "username": username,
                                    "password": password,
                                    "status": "Cracked"
                                })
                        break
                    except subprocess.SubprocessError:
                        continue
                os.remove(chunk_file)
                return chunk_results

            # Parallel processing for CPU mode
            with mp.Pool(processes=num_processes) as pool:
                chunk_results = pool.starmap(process_chunk, [(chunk, i) for i, chunk in enumerate(hash_chunks)])
            for chunk_result in chunk_results:
                results.extend(chunk_result)

        logger.info(f"Offline cracking completed for {hash_file}. Found {len(results)} results.")
    except subprocess.TimeoutExpired:
        logger.error("Cracking timed out.")
        results.append({"target": hash_file, "mode": "Offline", "status": "Timeout"})
    except Exception as e:
        logger.error(f"Offline cracking failed: {str(e)}")
        results.append({"target": hash_file, "mode": "Offline", "status": f"Error: {str(e)}"})
    finally:
        if os.path.exists(pwd_file):
            os.remove(pwd_file)
    return results