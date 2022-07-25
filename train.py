from utils import NeuralNet
from argparse import ArgumentParser
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    home_dir = os.getcwd()

    nn = NeuralNet(hidden_size=[50, 50], batch_size=32, lr=0.1, epochs=10)
    parser.add_argument(
        '--hidden_size', default=[50, 50, 50], nargs='+', required=True, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.01, type=float, required=True)
    parser.add_argument('--epochs', default=10, type=int, required=True)
    parser.add_argument('--activation_function', default='sigmoid', type=str)

    args = parser.parse_args()
    print('---------------------Neural Network with Numpy-------------------')
    print('The hyperparameter of the training')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))

    nn = NeuralNet(hidden_size=args.hidden_size,
                   batch_size=args.batch_size,
                   lr=args.lr,
                   epochs=args.epochs,
                   activation_function=args.activation_function)

    x_train, y_train, x_val, y_val = nn.getMNIST()
    A = nn.train(x_train, y_train, x_val, y_val)
