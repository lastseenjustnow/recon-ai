
# ATM Video Simulation and Synthetic Data Generator
# For hackathon demonstration without actual video files

import json
import random
import datetime
from typing import List, Dict, Any
import pandas as pd
import numpy as np

class ATMVideoSimulator:
    """
    Simulates video metadata and events for ATM cash deposit scenarios
    This creates synthetic data that mimics what computer vision would extract from real video
    """
    
    def __init__(self):
        self.behaviors = {
            'normal': {
                'face_visible': True,
                'person_count': 1,
                'duration_seconds': (60, 180),
                'suspicious_score': 0.1,
                'actions': ['approach', 'insert_card', 'enter_pin', 'insert_cash', 'confirm', 'take_card', 'leave']
            },
            'suspicious': {
                'face_visible': False,
                'person_count': (1, 3),
                'duration_seconds': (30, 60),
                'suspicious_score': 0.8,
                'actions': ['approach', 'look_around', 'cover_keypad', 'multiple_attempts', 'quick_transaction', 'leave']
            },
            'fraud': {
                'face_visible': False,
                'person_count': (2, 4),
                'duration_seconds': (20, 45),
                'suspicious_score': 0.95,
                'actions': ['approach', 'hood_up', 'multiple_cards', 'rapid_deposits', 'lookout_behavior', 'flee']
            }
        }
    
    def generate_video_metadata(self, 
                               scenario: str = 'normal',
                               atm_id: str = 'ATM_001',
                               timestamp: datetime.datetime = None) -> Dict[str, Any]:
        """
        Generate video metadata that would be extracted from actual surveillance footage
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        behavior = self.behaviors[scenario]
        
        # Calculate duration
        if isinstance(behavior['duration_seconds'], tuple):
            duration = random.randint(*behavior['duration_seconds'])
        else:
            duration = behavior['duration_seconds']
        
        # Generate frame-by-frame annotations
        frames = self._generate_frame_annotations(behavior, duration)
        
        # Generate person detection data
        if isinstance(behavior['person_count'], tuple):
            person_count = random.randint(*behavior['person_count'])
        else:
            person_count = behavior['person_count']
        
        video_data = {
            'video_id': f"VID_{timestamp.strftime('%Y%m%d_%H%M%S')}_{atm_id}",
            'atm_id': atm_id,
            'timestamp': timestamp.isoformat(),
            'duration_seconds': duration,
            'scenario_type': scenario,
            'person_detections': {
                'count': person_count,
                'face_visible': behavior['face_visible'],
                'confidence': random.uniform(0.85, 0.99)
            },
            'behavioral_analysis': {
                'suspicious_score': behavior['suspicious_score'] + random.uniform(-0.1, 0.1),
                'anomaly_detected': scenario != 'normal',
                'actions_sequence': behavior['actions']
            },
            'frame_annotations': frames,
            'environment': {
                'lighting': random.choice(['good', 'moderate', 'poor']),
                'camera_angle': random.choice(['front', 'side', 'overhead']),
                'obstruction': random.choice(['none', 'partial', 'significant']) if scenario == 'fraud' else 'none'
            }
        }
        
        return video_data
    
    def _generate_frame_annotations(self, behavior: Dict, duration: int) -> List[Dict]:
        """
        Generate per-frame annotations simulating computer vision output
        """
        fps = 30  # Standard surveillance camera FPS
        total_frames = duration * fps
        key_frames = []
        
        # Distribute actions across timeline
        action_points = np.linspace(0, total_frames-1, len(behavior['actions'])).astype(int)
        
        for i, (frame_num, action) in enumerate(zip(action_points, behavior['actions'])):
            key_frames.append({
                'frame_number': int(frame_num),
                'timestamp_offset': frame_num / fps,
                'action': action,
                'objects_detected': self._get_objects_for_action(action),
                'pose_estimation': self._get_pose_for_action(action),
                'confidence': random.uniform(0.7, 0.95)
            })
        
        return key_frames
    
    def _get_objects_for_action(self, action: str) -> List[str]:
        """
        Return objects that would be detected for each action
        """
        objects_map = {
            'approach': ['person', 'atm_machine'],
            'insert_card': ['person', 'hand', 'card', 'atm_slot'],
            'enter_pin': ['person', 'hand', 'keypad'],
            'insert_cash': ['person', 'hand', 'cash_bundle', 'deposit_slot'],
            'confirm': ['person', 'hand', 'screen'],
            'take_card': ['person', 'hand', 'card'],
            'leave': ['person'],
            'look_around': ['person', 'head_turn'],
            'cover_keypad': ['person', 'hand_over_keypad', 'obstruction'],
            'multiple_attempts': ['person', 'multiple_cards', 'repeated_motion'],
            'hood_up': ['person', 'face_obstruction', 'hood'],
            'multiple_cards': ['person', 'card_stack', 'switching_cards'],
            'rapid_deposits': ['person', 'cash_bundles', 'quick_motion'],
            'lookout_behavior': ['multiple_persons', 'watching_behavior'],
            'flee': ['person', 'running', 'quick_exit']
        }
        return objects_map.get(action, ['person'])
    
    def _get_pose_for_action(self, action: str) -> Dict[str, Any]:
        """
        Generate pose estimation data for actions
        """
        return {
            'body_facing': random.choice(['atm', 'sideways', 'away']),
            'head_position': random.choice(['forward', 'looking_around', 'down']),
            'arm_position': random.choice(['extended', 'at_side', 'covering']),
            'stance': random.choice(['normal', 'tense', 'relaxed'])
        }

class ATMTransactionGenerator:
    """
    Generate synthetic ATM transaction logs with cash deposit focus
    """
    
    def __init__(self):
        self.response_codes = {
            '000': 'Success',
            '001': 'Insufficient funds',
            '002': 'Invalid PIN',
            '003': 'Card expired',
            '004': 'Daily limit exceeded',
            '005': 'Technical error',
            '006': 'Suspected fraud',
            '007': 'Card retained',
            '008': 'Network timeout'
        }
        
        self.error_messages = {
            'SUCCESS': 'Transaction completed successfully',
            'DEPOSIT_MISMATCH': 'Counted amount differs from declared amount',
            'BILL_REJECTED': 'One or more bills rejected',
            'ENVELOPE_JAM': 'Deposit envelope jammed',
            'TIMEOUT': 'Transaction timeout',
            'CANCELLED': 'Transaction cancelled by user',
            'FRAUD_ALERT': 'Suspicious pattern detected'
        }
    
    def generate_deposit_transaction(self, 
                                    scenario: str = 'normal',
                                    timestamp: datetime.datetime = None) -> Dict[str, Any]:
        """
        Generate a cash deposit transaction log entry
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        # Generate transaction details based on scenario
        if scenario == 'normal':
            amount = random.choice([100, 200, 300, 500, 1000, 1500])
            response_code = '000'
            error_message = 'SUCCESS'
            bill_count = amount // 100
            account_entered = False
        elif scenario == 'suspicious':
            amount = random.choice([2000, 3000, 5000, 7500])
            response_code = random.choice(['000', '006'])
            error_message = random.choice(['SUCCESS', 'DEPOSIT_MISMATCH', 'FRAUD_ALERT'])
            bill_count = amount // 100
            account_entered = random.choice([True, False])
        else:  # fraud
            amount = random.choice([9000, 9500, 9900, 10000])  # Just under reporting limits
            response_code = random.choice(['000', '006', '007'])
            error_message = random.choice(['FRAUD_ALERT', 'SUCCESS'])
            bill_count = amount // 100
            account_entered = True
        
        transaction = {
            'transaction_id': f"TXN_{timestamp.strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            'transaction_type': 'CASH_DEPOSIT',
            'timestamp': timestamp.isoformat(),
            'machine_id': f"ATM_{random.randint(1, 50):03d}",
            'card_number_masked': f"****-****-****-{random.randint(1000, 9999)}",
            'account_selected': random.choice(['CHECKING', 'SAVINGS']) if not account_entered else 'MANUAL_ENTRY',
            'account_number': f"{'*' * 6}{random.randint(1000, 9999)}" if account_entered else None,
            'amount_declared': amount,
            'amount_counted': amount if error_message != 'DEPOSIT_MISMATCH' else amount - random.randint(20, 100),
            'bill_count': bill_count,
            'bill_denominations': self._generate_bill_breakdown(amount),
            'response_code': response_code,
            'response_description': self.response_codes[response_code],
            'error_message': self.error_messages[error_message],
            'user_actions': self._generate_user_actions(scenario),
            'session_duration_seconds': random.randint(60, 300) if scenario == 'normal' else random.randint(30, 90),
            'retry_count': 0 if scenario == 'normal' else random.randint(1, 3),
            'previous_failed_attempts': 0 if scenario == 'normal' else random.randint(0, 2)
        }
        
        return transaction
    
    def _generate_bill_breakdown(self, amount: int) -> Dict[str, int]:
        """
        Generate breakdown of bill denominations
        """
        denominations = {
            '100': 0,
            '50': 0,
            '20': 0,
            '10': 0,
            '5': 0
        }
        
        remaining = amount
        
        # Prefer larger bills
        for denom in [100, 50, 20, 10, 5]:
            if remaining >= denom:
                count = remaining // denom
                if random.random() > 0.3:  # 70% chance to use this denomination
                    use_count = random.randint(1, min(count, 20))
                    denominations[str(denom)] = use_count
                    remaining -= use_count * denom
        
        # Adjust to match total
        if remaining > 0:
            denominations['20'] += remaining // 20
            remaining = remaining % 20
            if remaining > 0:
                denominations['10'] += remaining // 10
                remaining = remaining % 10
                if remaining > 0:
                    denominations['5'] += remaining // 5
        
        return {k: v for k, v in denominations.items() if v > 0}
    
    def _generate_user_actions(self, scenario: str) -> List[str]:
        """
        Generate sequence of user actions during transaction
        """
        normal_actions = [
            'CARD_INSERTED',
            'PIN_ENTERED',
            'DEPOSIT_SELECTED',
            'AMOUNT_ENTERED',
            'CASH_INSERTED',
            'CONFIRM_PRESSED',
            'RECEIPT_PRINTED',
            'CARD_EJECTED'
        ]
        
        suspicious_actions = normal_actions + random.sample([
            'CANCEL_PRESSED',
            'TIMEOUT_WARNING',
            'AMOUNT_CORRECTED',
            'MULTIPLE_DEPOSITS'
        ], k=random.randint(1, 2))
        
        fraud_actions = [
            'CARD_INSERTED',
            'PIN_ENTERED_MULTIPLE',
            'QUICK_DEPOSIT',
            'MAX_AMOUNT',
            'NO_RECEIPT',
            'RAPID_COMPLETION'
        ]
        
        if scenario == 'normal':
            return normal_actions
        elif scenario == 'suspicious':
            return suspicious_actions
        else:
            return fraud_actions

def generate_hackathon_dataset(num_transactions: int = 5) -> tuple:
    """
    Generate a complete dataset for hackathon demonstration
    Returns both transaction logs and video metadata
    """
    video_sim = ATMVideoSimulator()
    trans_gen = ATMTransactionGenerator()
    
    transactions = []
    video_metadata = []
    
    # Generate mix of scenarios
    scenarios = ['normal'] * (num_transactions // 2) + \
                ['suspicious'] * (num_transactions // 3) + \
                ['fraud'] * (num_transactions - num_transactions // 2 - num_transactions // 3)
    
    random.shuffle(scenarios)
    
    for i, scenario in enumerate(scenarios):
        # Generate timestamp
        base_time = datetime.datetime.now() - datetime.timedelta(hours=random.randint(1, 72))
        timestamp = base_time + datetime.timedelta(minutes=random.randint(0, 60))
        
        # Generate ATM ID
        atm_id = f"ATM_{random.randint(1, 10):03d}"
        
        # Generate transaction
        transaction = trans_gen.generate_deposit_transaction(scenario, timestamp)
        transaction['scenario'] = scenario  # Add label for training
        transactions.append(transaction)
        
        # Generate corresponding video metadata
        video = video_sim.generate_video_metadata(scenario, atm_id, timestamp)
        video_metadata.append(video)
        
        print(f"Generated {scenario} scenario {i+1}/{num_transactions}")
    
    # Convert to DataFrames for easy manipulation
    trans_df = pd.DataFrame(transactions)
    video_df = pd.DataFrame(video_metadata)
    
    return trans_df, video_df

def save_synthetic_data():
    """
    Generate and save synthetic data for hackathon
    """
    print("Generating synthetic ATM fraud detection dataset...")
    print("-" * 50)
    
    # Generate 5 transactions as requested
    trans_df, video_df = generate_hackathon_dataset(5)
    
    # Save to CSV
    trans_df.to_csv('data/atm_transactions.csv', index=False)
    print(f"✓ Saved {len(trans_df)} transactions to 'atm_transactions.csv'")
    
    # Save video metadata as JSON for easier parsing
    video_data = video_df.to_dict('records')
    with open('config/video_metadata.json', 'w') as f:
        json.dump(video_data, f, indent=2)
    print(f"✓ Saved {len(video_df)} video metadata records to 'video_metadata.json'")
    
    # Display sample data
    print("\n" + "="*50)
    print("SAMPLE TRANSACTION (First Fraud Case):")
    print("="*50)
    fraud_trans = trans_df[trans_df['scenario'] == 'fraud'].iloc[0] if len(trans_df[trans_df['scenario'] == 'fraud']) > 0 else trans_df.iloc[0]
    for key, value in fraud_trans.items():
        if key != 'scenario':
            print(f"{key:25s}: {value}")
    
    print("\n" + "="*50)
    print("CORRESPONDING VIDEO METADATA:")
    print("="*50)
    fraud_video = video_df[video_df['scenario_type'] == 'fraud'].iloc[0] if len(video_df[video_df['scenario_type'] == 'fraud']) > 0 else video_df.iloc[0]
    print(f"Video ID: {fraud_video['video_id']}")
    print(f"Duration: {fraud_video['duration_seconds']} seconds")
    print(f"Person Count: {fraud_video['person_detections']['count']}")
    print(f"Face Visible: {fraud_video['person_detections']['face_visible']}")
    print(f"Suspicious Score: {fraud_video['behavioral_analysis']['suspicious_score']:.2f}")
    print(f"Actions Detected: {', '.join(fraud_video['behavioral_analysis']['actions_sequence'])}")
    
    print("\n" + "="*50)
    print("DATASET SUMMARY:")
    print("="*50)
    print(f"Total Transactions: {len(trans_df)}")
    print(f"- Normal: {len(trans_df[trans_df['scenario'] == 'normal'])}")
    print(f"- Suspicious: {len(trans_df[trans_df['scenario'] == 'suspicious'])}")
    print(f"- Fraud: {len(trans_df[trans_df['scenario'] == 'fraud'])}")
    print(f"\nTotal Amount Deposited: ${trans_df['amount_declared'].sum():,}")
    print(f"Average Transaction: ${trans_df['amount_declared'].mean():.2f}")
    print(f"Fraud Detection Rate: {len(trans_df[trans_df['response_code'] == '006']) / len(trans_df) * 100:.1f}%")

# Integration with Azure OpenAI for fraud detection
class FraudDetectionDemo:
    """
    Demo class showing how to use the synthetic data with Azure OpenAI
    """
    
    def analyze_transaction_with_video(self, transaction: Dict, video: Dict) -> Dict:
        """
        Analyze transaction with video context for fraud detection
        """
        # Create prompt combining transaction and video data
        prompt = f"""
        Analyze this ATM cash deposit for potential fraud:
        
        TRANSACTION DATA:
        - Amount: ${transaction['amount_declared']}
        - Time: {transaction['timestamp']}
        - Machine: {transaction['machine_id']}
        - Bill Count: {transaction['bill_count']}
        - Response Code: {transaction['response_code']}
        - User Actions: {transaction['user_actions']}
        - Session Duration: {transaction['session_duration_seconds']} seconds
        
        VIDEO ANALYSIS:
        - Persons Detected: {video['person_detections']['count']}
        - Face Visible: {video['person_detections']['face_visible']}
        - Suspicious Score: {video['behavioral_analysis']['suspicious_score']}
        - Behavior Sequence: {video['behavioral_analysis']['actions_sequence']}
        - Environment: {video['environment']}
        
        Provide:
        1. Fraud probability (0-100%)
        2. Key risk indicators
        3. Recommended action
        """
        
        # This would connect to Azure OpenAI in production
        # For demo, return mock analysis
        return {
            'fraud_probability': video['behavioral_analysis']['suspicious_score'] * 100,
            'risk_indicators': [
                'Multiple persons present' if video['person_detections']['count'] > 1 else None,
                'Face not visible' if not video['person_detections']['face_visible'] else None,
                'Large deposit amount' if transaction['amount_declared'] > 5000 else None,
                'Quick transaction' if transaction['session_duration_seconds'] < 60 else None
            ],
            'recommended_action': 'Flag for review' if video['behavioral_analysis']['suspicious_score'] > 0.5 else 'Approve',
            'prompt_used': prompt
        }
