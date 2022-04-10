import argparse

from train import prepare_trainer


# args
parser = argparse.ArgumentParser(description='tab-transformer trainer parser')
parser.add_argument('--config_file', '-c', type=str, default='v3', help='config file name')
args = parser.parse_args()


def main():
    trainer, learner, train_loader, val_loader = prepare_trainer(args.config_file)
    trainer.fit(learner, train_loader, val_loader)


if __name__ == '__main__':
    main()
