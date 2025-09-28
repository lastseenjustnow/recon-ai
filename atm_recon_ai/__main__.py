# Load your generated data
import pandas as pd
import json

from atm_recon_ai.atm_recon_ai import save_synthetic_data

def run_save_synthetic_data():
    # Generate and save the synthetic dataset
    save_synthetic_data()

    print("\n" + "=" * 50)
    print("HACKATHON READY!")
    print("=" * 50)
    print("✓ Transaction logs saved to: atm_transactions.csv")
    print("✓ Video metadata saved to: video_metadata.json")
    print("\nThese files simulate what computer vision would extract from real ATM videos.")
    print("Use them to demonstrate your fraud detection system without actual video files!")
    print("\nNext steps:")
    print("1. Load these files in your fraud detection system")
    print("2. Use the video metadata as if it came from video analysis")
    print("3. Combine transaction + video data for comprehensive fraud detection")
    print("4. Show how suspicious patterns in video correlate with transaction anomalies")

    # Load transaction and video data
    transactions = pd.read_csv('./data/atm_transactions.csv')
    with open('./config/video_metadata.json', 'r') as f:
        videos = json.load(f)

    # Show fraud detection in action
    for trans, video in zip(transactions.to_dict('records'), videos):
        if video['behavioral_analysis']['suspicious_score'] > 0.7:
            print(f"🚨 ALERT: Suspicious activity detected!")
            print(f"   Transaction: ${trans['amount_declared']}")
            print(f"   Video: {video['person_detections']['count']} people, face hidden")
            print(f"   Action: {trans['response_code']} - {trans['error_message']}")

if __name__ == "__main__":
     from atm_recon_ai.ollama import call_ollama
     call_ollama.run()