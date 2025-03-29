# main.py

import argparse
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Cyber Threat Detection Models")
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    args = parser.parse_args()
    
    train()
