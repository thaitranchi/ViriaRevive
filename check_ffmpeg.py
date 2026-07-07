from hwaccel import probe_ffmpeg, resolve_yolo_device, resolve_whisper_device
import shutil
import sys

def check_hardware_support():
    print(f"Python: {sys.executable}")
    if sys.prefix == sys.base_prefix:
        print("[!] WARNING: Not running in a virtual environment. Activate venv first:")
        print("    .\\venv\\Scripts\\Activate")
        print()
    else:
        print("[+] Virtual environment detected")

    print("[*] Probing FFmpeg for hardware support...")
    prof = probe_ffmpeg()
    
    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")
    
    print(f"FFmpeg Path:  {ffmpeg_bin or 'NOT FOUND'}")
    print(f"FFprobe Path: {ffprobe_bin or 'NOT FOUND'}")

    print(f"\nAvailable Accelerators: {', '.join(prof.hwaccels)}")
    print(f"Available Encoders:     {', '.join(prof.encoders)}")
    
    if 'cuda' in prof.hwaccels:
        print("\n[+] SUCCESS: Your FFmpeg build supports 'cuda' acceleration.")
    else:
        print("\n[!] WARNING: 'cuda' is not supported in FFmpeg. Check your build or drivers.")

    print("\n" + "="*40)
    print("[*] Probing PyTorch (YOLO/Whisper) support...")
    try:
        import torch
        print(f"Torch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_avail}")
        
        if cuda_avail:
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("\n[!] WARNING: PyTorch cannot see your GPU.")
            print("    To fix this, run:")
            print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
            print("    Or (with requirements.txt):")
            print("    pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124")
            
    except ImportError:
        print("\n[!] FAILURE: 'torch' is not installed.")
        print("    Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        print("    Or:  pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124")

    yolo = resolve_yolo_device("cuda")
    whisper_dev, whisper_compute, whisper_idx = resolve_whisper_device("cuda")
    print(f"\nResolved YOLO Device:    {yolo}")
    print(f"Resolved Whisper Device: {whisper_dev}/{whisper_compute} (idx={whisper_idx})")
    print("="*40)

if __name__ == "__main__":
    check_hardware_support()
