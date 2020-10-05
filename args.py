import argparse
import os


def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="NELL", type=str)
    parser.add_argument("--embed_dim", default=100, type=int)
    parser.add_argument("--train_few", default=5, type=int)
    parser.add_argument("--few", default=5, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--neg_num", default=1, type=int)
    parser.add_argument("--random_embed", action='store_true')
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--margin", default=5.0, type=float)
    parser.add_argument("--dropout_input", default=0.3, type=float)
    parser.add_argument("--dropout_layers", default=0.2, type=float)
    parser.add_argument("--dropout_neighbors", default=0.0, type=float)

    parser.add_argument("--process_steps", default=2, type=int)
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--eval_every", default=10000, type=int)
    parser.add_argument("--fine_tune", action='store_true')
    parser.add_argument("--max_neighbor", default=50, type=int)
    parser.add_argument("--no_meta", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--dev", action='store_true')
    parser.add_argument("--embed_model", default='TransE', type=str)
    parser.add_argument("--prefix", default='intial', type=str)

    parser.add_argument("--seed", default='19950922', type=int)
    parser.add_argument("--loss", default='origin', type=str)

    parser.add_argument("--num_transformer_layers", default=3, type=int)
    parser.add_argument("--num_transformer_heads", default=4, type=int)
    parser.add_argument("--warm_up_step", default=10000, type=int)
    parser.add_argument("--max_batches", default=300000, type=int)
    parser.add_argument("--weight_decay", default=0.0001, type=float)
    parser.add_argument("--grad_clip", default=5.0, type=float)

    args = parser.parse_args()
    if not os.path.exists('models'):
        os.mkdir('models')
    args.save_path = 'models/' + args.prefix

    return args


if __name__ == '__main__':
    args = read_options()
