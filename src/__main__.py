"""FedRNK: Federate Reasoning, Not Knowledge

Usage:
    python -m src.train --experiment e0 --method fedrnk
    python -m src.analysis.gradient_analysis --experiment e0
"""

from src.train import main

if __name__ == "__main__":
    main()
