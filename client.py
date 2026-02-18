import sounddevice as sd
import numpy as np
import opuslib
import socket
import threading
import sys
import time
import opuslib.api.encoder as encoder_api

# ==========================================================
# --- OPUS CONSTANTS (from opus_defines.h) ---
# These are the official C API control constants.
# ==========================================================
OPUS_SET_APPLICATION_REQUEST = 4000
OPUS_GET_APPLICATION_REQUEST = 4001
OPUS_SET_BITRATE_REQUEST = 4002            # Set encoder target bitrate (bps)
OPUS_GET_BITRATE_REQUEST = 4003
OPUS_SET_VBR_REQUEST = 4006                # Enable/disable VBR
OPUS_GET_VBR_REQUEST = 4007
OPUS_SET_COMPLEXITY_REQUEST = 4010         # Set encoder complexity (0–10)
OPUS_GET_COMPLEXITY_REQUEST = 4011
OPUS_SET_INBAND_FEC_REQUEST = 4012         # Enable/disable in-band FEC
OPUS_GET_INBAND_FEC_REQUEST = 4013
OPUS_SET_PACKET_LOSS_PERC_REQUEST = 4014   # Expected packet loss percentage
OPUS_GET_PACKET_LOSS_PERC_REQUEST = 4015
OPUS_SET_DTX_REQUEST = 4016                # Enable/disable DTX (silence compression)
OPUS_GET_DTX_REQUEST = 4017

# ==========================================================
# --- AUDIO CONFIGURATION ---
# ==========================================================
SAMPLING_RATE = 16000     # 16 kHz sampling rate (narrowband/mid-quality VoIP)
CHANNELS = 1              # Mono audio
FRAME_DURATION_MS = 20    # 20ms per frame (standard for real-time VoIP)
FRAME_SIZE = (SAMPLING_RATE // 1000) * FRAME_DURATION_MS  # 320 samples/frame
RECORD_SECONDS = 1        # Record 1 second of audio per push-to-talk

# ==========================================================
# --- GLOBAL ENCODER STATE ---
# ==========================================================
CURRENT_BITRATE = 32000       # 32 kbps
CURRENT_USE_FEC = False       # Forward Error Correction disabled by default
CURRENT_COMPLEXITY = 5        # Opus complexity (0–10)
encoder = None                # Will be initialized in main()

# ==========================================================
# --- FUNCTIONS ---
# ==========================================================
def set_encoder_settings(bitrate: int, use_fec: bool, complexity: int):
    global CURRENT_BITRATE, CURRENT_USE_FEC, CURRENT_COMPLEXITY
    try:
        encoder_api.encoder_ctl(encoder, OPUS_SET_BITRATE_REQUEST, bitrate)
        encoder_api.encoder_ctl(encoder, OPUS_SET_INBAND_FEC_REQUEST, 1 if use_fec else 0)
        encoder_api.encoder_ctl(encoder, OPUS_SET_COMPLEXITY_REQUEST, complexity)

        CURRENT_BITRATE = bitrate
        CURRENT_USE_FEC = use_fec
        CURRENT_COMPLEXITY = complexity
        return True
    except Exception as e:
        print(f"Error setting encoder settings: {e}")
        return False



def audio_receiver(listen_port: int):
    """
    Background thread that listens for UDP packets,
    decodes them, and plays them live.
    """
    try:
        decoder = opuslib.Decoder(SAMPLING_RATE, CHANNELS)
        stream = sd.OutputStream(
            samplerate=SAMPLING_RATE, 
            channels=CHANNELS, 
            dtype='int16'
        )
        stream.start()
    except Exception as e:
        print(f"Receiver Error: Failed to init audio stream: {e}")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(('0.0.0.0', listen_port))
    except OSError as e:
        print(f"Receiver Error: Port {listen_port} is already in use. {e}")
        return
        
    print(f"\n[Receiver] Listening on port {listen_port}...")

    while True:
        try:
            packet, addr = sock.recvfrom(1024)
            pcm_data = decoder.decode(packet, FRAME_SIZE, decode_fec=True)
            stream.write(pcm_data)
        except opuslib.OpusError:
            # Ignore minor decode errors (e.g., lost/corrupt packets)
            pass
        except Exception as e:
            print(f"Receiver Error: {e}")


def main(my_port: int, dest_port: int):
    """
    The main Push-to-Talk function.
    """
    global encoder
    
    try:
        encoder = opuslib.Encoder(SAMPLING_RATE, CHANNELS, opuslib.APPLICATION_VOIP)
        set_encoder_settings(CURRENT_BITRATE, CURRENT_USE_FEC, CURRENT_COMPLEXITY)
    except opuslib.OpusError as e:
        print(f"Error creating Opus encoder: {e}")
        sys.exit(1)
    
    # Start the receiver in the background
    receiver_thread = threading.Thread(
        target=audio_receiver, 
        args=(my_port,), 
        daemon=True
    )
    receiver_thread.start()

    # Set up UDP sending socket
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest_address = ('127.0.0.1', dest_port)
    
    print(f"\n[Sender] Ready. Sending to port {dest_port}.")
    print("Press ENTER to talk for 1 second...")

    while True:
        try:
            input()  # Wait for Enter key
            print(f"Recording... (Bitrate: {CURRENT_BITRATE}bps, FEC: {CURRENT_USE_FEC})")

            try:
                audio_data = sd.rec(
                    frames=int(RECORD_SECONDS * SAMPLING_RATE),
                    samplerate=SAMPLING_RATE,
                    channels=CHANNELS,
                    dtype='int16'
                )
                sd.wait()
            except sd.PortAudioError as e:
                print(f"Error: No audio input device found. {e}")
                print("Generating 1s of synthetic noise to test the pipeline...")
                audio_data = (np.random.rand(int(RECORD_SECONDS * SAMPLING_RATE), 1) * 1000).astype('int16')

            # Send audio in 20ms frames
            num_frames = len(audio_data) // FRAME_SIZE
            for i in range(num_frames):
                start = i * FRAME_SIZE
                end = start + FRAME_SIZE
                pcm_frame = audio_data[start:end].tobytes()

                encoded_packet = encoder.encode(pcm_frame, FRAME_SIZE)
                send_sock.sendto(encoded_packet, dest_address)

            print("...Sending complete.")
            print("\nPress ENTER to talk for 1 second...")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Main Loop Error: {e}")


# ==========================================================
# --- ENTRY POINT ---
# ==========================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 client.py <my_port> <destination_port>")
        print("Example: python3 client.py 8000 9000")
        sys.exit(1)

    try:
        MY_PORT = int(sys.argv[1])
        DEST_PORT = int(sys.argv[2])
        main(MY_PORT, DEST_PORT)
    except ValueError:
        print("Error: Ports must be numbers.")
        sys.exit(1)
