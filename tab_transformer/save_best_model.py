import os
import argparse
import wandb

from config import get_cfg_defaults


# args
parser = argparse.ArgumentParser(description='tab-transformer trainer parser')
parser.add_argument('--description', '-d', type=str, default='', help='best model description')
args = parser.parse_args()


def main():
    # references
    # 1) https://docs.wandb.ai/ref/python/artifact
    # 2) https://docs.wandb.ai/guides/artifacts/api
    cfg = get_cfg_defaults()

    # Save as artifact for version control.
    run = wandb.init(project='tab-transformer-experiment')

    # TODO: insert metadata (dict)
    artifact = wandb.Artifact(
        name='TabTransformer-Best-Model',
        type='model',
        description=f'{args.description}',
        metadata=None,
    )
    artifact.add_file(os.path.join(cfg.ADDRESS.CHECK, 'best_model.ckpt'))

    run.log_artifact(artifact)
    run.finish()


if __name__ == '__main__':
    main()
