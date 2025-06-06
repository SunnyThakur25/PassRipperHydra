# main.py
import streamlit as st
import pandas as pd
import os
import logging
import time
import plotly.express as px
from core.passgan import PassGANModel
from core.john_wrapper import crack_hashes
from core.hydra_wrapper import brute_force_online
from utils.helpers import sanitize_input, load_logo, solve_captcha, rotate_proxy, get_random_user_agent
from utils.logger import setup_logging
from utils.exporter import export_results

# Setup logging
logger = setup_logging("output/logs/passripperhydra.log")

# Initialize PassGAN with advanced configuration
if "passgan_config" not in st.session_state:
    st.session_state.passgan_config = {
        "charset": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?",
        "min_len": 6,
        "max_len": 12,
        "min_complexity": True
    }

passgan = PassGANModel(
    weights_path="core/passgan_weights.pkl",
    checkpoint_path="fine_tuned/checkpoints/checkpoint_200000.ckpt",
    charset=st.session_state.passgan_config["charset"],
    min_len=st.session_state.passgan_config["min_len"],
    max_len=st.session_state.passgan_config["max_len"]
)

# Streamlit page setup
st.set_page_config(page_title="PassRipperHydra", layout="wide", initial_sidebar_state="expanded")

# Theme toggle state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Inject Tailwind CSS, custom cyberpunk styling, and theme toggle
theme_colors = {
    "dark": {
        "background": "linear-gradient(135deg, #0d1b2a, #1b263b)",
        "text": "#e0e0e0",
        "input_bg": "#2a2a4a",
        "input_border": "#ff4b4b",
        "neon": "#ff4b4b",
        "glow": "#00ffcc",
        "button_bg": "linear-gradient(45deg, #ff00ff, #00ffcc)",
        "button_hover_bg": "linear-gradient(45deg, #00ffcc, #ff00ff)",
        "panel_bg": "rgba(42, 42, 74, 0.8)",
        "panel_border": "#00ffcc",
    },
    "light": {
        "background": "linear-gradient(135deg, #e0e0e0, #ffffff)",
        "text": "#0d1b2a",
        "input_bg": "#ffffff",
        "input_border": "#ff4b4b",
        "neon": "#ff4b4b",
        "glow": "#00ffcc",
        "button_bg": "linear-gradient(45deg, #ff4b4b, #00ffcc)",
        "button_hover_bg": "linear-gradient(45deg, #00ffcc, #ff4b4b)",
        "panel_bg": "rgba(255, 255, 255, 0.9)",
        "panel_border": "#ff4b4b",
    }
}

current_theme = theme_colors[st.session_state.theme]
st.markdown(f"""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        @keyframes neon-glow {{
            0% {{ text-shadow: 0 0 5px {current_theme['neon']}, 0 0 10px {current_theme['neon']}, 0 0 15px {current_theme['glow']}; }}
            50% {{ text-shadow: 0 0 10px {current_theme['neon']}, 0 0 20px {current_theme['neon']}, 0 0 30px {current_theme['glow']}; }}
            100% {{ text-shadow: 0 0 5px {current_theme['neon']}, 0 0 10px {current_theme['neon']}, 0 0 15px {current_theme['glow']}; }}
        }}
        @keyframes pulse {{
            0% {{ opacity: 0.8; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0.8; }}
        }}
        body {{
            background: {current_theme['background']};
            color: {current_theme['text']};
            font-family: 'Arial', sans-serif;
        }}
        .stApp {{
            background: transparent;
        }}
        .neon-text {{
            animation: neon-glow 2s infinite;
            color: {current_theme['neon']};
        }}
        .holographic-panel {{
            background: {current_theme['panel_bg']};
            border: 2px solid {current_theme['panel_border']};
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 0 15px rgba(0, 255, 204, 0.5);
        }}
        .cyber-button {{
            background: {current_theme['button_bg']};
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
            transition: all 0.3s;
            animation: pulse 2s infinite;
        }}
        .cyber-button:hover {{
            background: {current_theme['button_hover_bg']};
            box-shadow: 0 0 10px {current_theme['neon']};
        }}
        .stTextInput > div > div > input {{
            background-color: {current_theme['input_bg']};
            color: {current_theme['text']};
            border: 2px solid {current_theme['input_border']};
            border-radius: 5px;
        }}
        .stSelectbox > div > div > select {{
            background-color: {current_theme['input_bg']};
            color: {current_theme['text']};
            border: 2px solid {current_theme['input_border']};
            border-radius: 5px;
        }}
        .stTable, .stDataFrame {{
            background-color: {current_theme['input_bg']};
            border: 1px solid {current_theme['panel_border']};
        }}
        .stImage > img {{
            border: 3px solid {current_theme['neon']};
            border-radius: 10px;
            box-shadow: 0 0 20px {current_theme['glow']}, 0 0 30px {current_theme['neon']};
            transition: transform 0.3s ease-in-out;
        }}
        .stImage > img:hover {{
            transform: scale(1.1);
        }}
    </style>
""", unsafe_allow_html=True)

# Sidebar: Configuration
with st.sidebar:
    # Logo display
    st.image(load_logo("assets/logo.png"), width=200, caption="PassRipperHydra")

    # Theme toggle
    theme_option = st.selectbox("Theme", ["Dark", "Light"], index=0 if st.session_state.theme == "dark" else 1)
    if theme_option.lower() != st.session_state.theme:
        st.session_state.theme = theme_option.lower()
        st.experimental_rerun()

    st.markdown('<h2 class="neon-text text-lg font-bold">Attack Configuration</h2>', unsafe_allow_html=True)

    # Attack Mode
    mode = st.selectbox("Select Mode", ["Offline (Hash Cracking)", "Online (Brute-Force)", "Password Generation"])
    
    # Target Input
    if mode == "Offline (Hash Cracking)":
        hash_file_input = st.file_uploader("Upload Hash File (e.g., hashes.txt):", type=["txt"], key="hash_file")
        use_gpu = st.checkbox("Use GPU Acceleration", value=True, key="use_gpu")
        num_processes = st.slider("Number of Processes (CPU Mode)", 1, 8, 4, key="num_processes")
    elif mode == "Online (Brute-Force)":
        target_url = st.text_input("Target URL (e.g., http://example.com):", key="target_url")
        service_type = st.selectbox("Service Type", ["SSH", "HTTP-Form", "FTP", "SMB", "RDP", "Telnet", "MySQL", "LDAP", "PostgreSQL", "SNMP", "SIP"], key="service_type")
        username = st.text_input("Username (or list path):", "admin", key="username")
    else:  # Password Generation
        num_samples = st.number_input("Number of Passwords to Generate", min_value=1, max_value=10000, value=1000, key="num_samples")

    # Metadata
    metadata_input = st.text_area("Metadata (e.g., usernames, keywords, one per line):", key="metadata")

    # PassGAN Configuration
    with st.expander("PassGAN Settings", expanded=False):
        charset = st.text_input("Character Set", st.session_state.passgan_config["charset"], key="charset")
        min_len = st.slider("Minimum Password Length", 4, 16, st.session_state.passgan_config["min_len"], key="min_len")
        max_len = st.slider("Maximum Password Length", min_len, 16, st.session_state.passgan_config["max_len"], key="max_len")
        min_complexity = st.checkbox("Enforce Minimum Complexity (digit + special char)", st.session_state.passgan_config["min_complexity"], key="min_complexity")
        if st.button("Apply PassGAN Settings"):
            st.session_state.passgan_config = {
                "charset": charset,
                "min_len": min_len,
                "max_len": max_len,
                "min_complexity": min_complexity
            }
            st.experimental_rerun()

    # Additional Settings
    proxy_enabled = st.checkbox("Enable Proxy Rotation", value=True, key="proxy_enabled")
    captcha_enabled = st.checkbox("Enable Advanced Captcha Solver", value=True, key="captcha_enabled")
    output_format = st.selectbox("Export Format", ["JSON", "CSV", "PDF"], key="output_format")

# Main Page
st.markdown('<h1 class="neon-text text-4xl font-bold text-center mb-6">⚡ PassRipperHydra ⚡</h1>', unsafe_allow_html=True)
st.markdown('<p class="text-center text-gray-400 mb-8">Advanced red team tool for password cracking with AI, offline, and online capabilities.</p>', unsafe_allow_html=True)

# Attack Launch Section
if st.button("Launch Attack", key="run_button", help="Start password cracking"):
    # Input Validation
    if mode == "Offline (Hash Cracking)" and not hash_file_input:
        st.error("Please upload a hash file!")
        logger.error("Missing hash file for offline attack.")
    elif mode == "Online (Brute-Force)" and not target_url:
        st.error("Please provide a target URL!")
        logger.error("Missing target URL for online attack.")
    else:
        try:
            # Sanitize Inputs
            if mode == "Offline (Hash Cracking)":
                hash_path = f"temp_{hash_file_input.name}"
                with open(hash_path, "wb") as f:
                    f.write(hash_file_input.getbuffer())
                hash_file = sanitize_input(hash_path, input_type="path")
            else:
                hash_file = None
            target_url = sanitize_input(target_url, input_type="url") if mode == "Online (Brute-Force)" else None
            username = sanitize_input(username) if mode == "Online (Brute-Force)" else None
            metadata = [sanitize_input(m.strip()) for m in metadata_input.split("\n") if m.strip()]

            # Progress Bar and Live Updates
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Initializing attack...")

            # Generate Passwords
            progress_bar.progress(10)
            status_text.text("Generating passwords with PassGAN...")
            passwords = passgan.generate(
                metadata=metadata,
                num_passwords=1000 if mode != "Password Generation" else num_samples,
                min_complexity=st.session_state.passgan_config["min_complexity"]
            )

            # Simulate live progress updates for password generation
            for i in range(10, 50, 5):
                progress_bar.progress(i)
                status_text.text(f"Generating passwords... {i}%")
                time.sleep(0.5)  # Simulate processing time

            # Run Attack
            progress_bar.progress(50)
            status_text.text("Executing attack...")
            results = []
            if mode == "Offline (Hash Cracking)":
                results = crack_hashes(hash_file, passwords, num_processes, use_gpu)
                os.remove(hash_file)  # Clean up temp file
            elif mode == "Online (Brute-Force)":
                results = brute_force_online(target_url, service_type, username, passwords, proxy_enabled, captcha_enabled)
            else:  # Password Generation
                results = [{"password": pwd} for pwd in passwords]

            # Simulate attack progress
            for i in range(50, 100, 5):
                progress_bar.progress(i)
                status_text.text(f"Executing attack... {i}%")
                time.sleep(0.5)  # Simulate processing time

            progress_bar.progress(100)
            status_text.text("Attack completed!")

            # Display Results
            st.markdown('<h2 class="neon-text text-2xl font-bold mt-8">Attack Results</h2>', unsafe_allow_html=True)
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # Visualize Results (if applicable)
                if mode != "Password Generation":
                    success_count = len([r for r in results if r.get("status") == "Cracked"])
                    failure_count = len(results) - success_count
                    if success_count or failure_count:
                        fig = px.pie(
                            values=[success_count, failure_count],
                            names=["Cracked", "Failed"],
                            title="Attack Success Rate",
                            color_discrete_sequence=["#00ffcc", "#ff4b4b"]
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Password Complexity Distribution (for Password Generation mode)
                if mode == "Password Generation":
                    lengths = [len(pwd["password"]) for pwd in results]
                    fig = px.histogram(
                        x=lengths,
                        nbins=10,
                        title="Password Length Distribution",
                        labels={"x": "Password Length", "y": "Count"},
                        color_discrete_sequence=["#ff00ff"]
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Export Results
                output_dir = "output/results/cracked_hashes" if mode == "Offline (Hash Cracking)" else \
                            "output/results/online_results" if mode == "Online (Brute-Force)" else \
                            "output/results/generated_passwords"
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/results_{time.strftime('%Y%m%d_%H%M%S')}"
                export_results(results, output_format.lower(), output_file)
                st.success(f"Results exported to {output_file}.{output_format.lower()}")
                logger.info(f"Attack completed. Results exported to {output_file}.{output_format.lower()}")

                # Download Button
                with open(f"{output_file}.{output_format.lower()}", "rb") as f:
                    st.download_button(
                        label="Download Results",
                        data=f,
                        file_name=f"results_{time.strftime('%Y%m%d_%H%M%S')}.{output_format.lower()}",
                        mime="application/octet-stream"
                    )
            else:
                st.warning("No results found.")
                logger.warning("Attack completed with no results found.")
        except Exception as e:
            st.error(f"Attack failed: {str(e)}. Please check the logs for details.")
            logger.error(f"Attack failed: {str(e)}")

# Real-Time Logs
with st.expander("View Real-Time Logs", expanded=True):
    log_file = "output/logs/passripperhydra.log"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = f.read()
        st.text_area("Logs", logs, height=300)
    else:
        st.info("No logs available yet.")

# Footer
st.markdown("---")
st.markdown('<p class="text-center text-gray-500">Developed by Red Team | Powered by xAI | June 2025</p>', unsafe_allow_html=True)