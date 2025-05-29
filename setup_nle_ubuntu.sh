#!/bin/bash

echo "Setting up NLE environment in Ubuntu..."
echo "========================================="

# Update system packages
echo "Step 1: Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and development tools
echo "Step 2: Installing Python and development tools..."
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential cmake ninja-build git curl wget

# Install additional dependencies that NLE needs
echo "Step 3: Installing NLE dependencies..."
sudo apt install -y bison flex libbz2-dev

# Create virtual environment
echo "Step 4: Creating virtual environment..."
python3 -m venv ~/nle-env

# Activate environment and install NLE
echo "Step 5: Installing NLE..."
source ~/nle-env/bin/activate
pip install --upgrade pip
pip install nle

# Test installation
echo "Step 6: Testing NLE installation..."
python3 -c "import nle; print('✓ NLE installed successfully!')"

# Install additional useful packages
echo "Step 7: Installing additional ML/RL packages..."
pip install torch gymnasium numpy matplotlib

echo ""
echo "========================================="
echo "✓ Setup complete!"
echo ""
echo "To use NLE in the future:"
echo "1. Open Ubuntu terminal"
echo "2. Run: source ~/nle-env/bin/activate"
echo "3. Then you can use Python with NLE"
echo ""
echo "Your Windows files are accessible at: /mnt/c/"
echo "Your project folder is at: /mnt/c/Users/enuya/OneDrive/Documents/GitHub/rl_minihack_jinadu/"
echo ""
echo "Example usage:"
echo "cd /mnt/c/Users/enuya/OneDrive/Documents/GitHub/rl_minihack_jinadu/"
echo "source ~/nle-env/bin/activate"
echo "python3 your_script.py"
echo "========================================="