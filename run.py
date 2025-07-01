import fire
import sys
from spatialnca.config import Config
from spatialnca.train import train


def main(**kwargs):
    cfg = Config(**kwargs)
    train(cfg, print_cfg=True)


if __name__ == "__main__":
    # Call Fire only if there actually are CLI arguments
    if len(sys.argv) > 1:
        fire.Fire(main)
    else:
        main()
