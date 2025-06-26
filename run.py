from spatialnca.config import Config
from spatialnca.train import train

if __name__ == "__main__":
    train(Config(), print_cfg=True)
