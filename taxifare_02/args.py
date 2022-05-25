import argparse
from sklearn.linear_model import SGDRegressor

# Create the argument parser for each parameter plus the job directory
parser = argparse.ArgumentParser()

parser.add_argument(
    '--job-dir',  # Handled automatically by AI Platform
    help='GCS location to write checkpoints and export models',
    required=True
    )
parser.add_argument(
    '--alpha',  # Specified in the config file
    help='Constant that multiplies the regularization term',
    default=0.0001,
    type=float
    )
parser.add_argument(
    '--max_iter',  # Specified in the config file
    help='Max number of iterations.',
    default=1000,
    type=int
    )
parser.add_argument(
    '--loss',  # Specified in the config file
    help='Loss function to be used',
    default='hinge',
    type=str
    )
parser.add_argument(
    '--penalty',  # Specified in the config file
    help='The penalty (aka regularization term) to be used',
    default='l2',
    type=str
    )

args = parser.parse_args()

model = SGDRegressor(
    alpha=args.alpha,
    max_iter=args.max_iter,
    loss=args.loss,
    penalty=args.penalty
    )

if __name__ == '__main__':
    print(args)
    print(model)