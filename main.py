from algorithms.ddpg.ddpg import run_ddpg
from algorithms.hjdqn.hjdqn import run_hjdqn

import argparse
import torch

parser = argparse.ArgumentParser()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser.add_argument('--algo', required=True, choices=['hjdqn', 'ddpg'])
parser.add_argument('--env', required=True)
parser.add_argument('--device', required=False, default=device)
parser.add_argument('--num_trial', required=False, default=1, type=int)
parser.add_argument('--max_iter', required=False, default=1e6, type=float)
parser.add_argument('--eval_interval', required=False, default=2000, type=int)
parser.add_argument('--render', action='store_true')
parser.add_argument('--L', required=False, default=30.0, type=float)
parser.add_argument('--tau', required=False, default=1e-3, type=float)
parser.add_argument('--lr', required=False, default=1e-4, type=float)
parser.add_argument('--noise', required=False, default='gaussian')
parser.add_argument('--sigma', required=False, default=0.1, type=float)
parser.add_argument('--hidden', required=False, default=256, type=int)
parser.add_argument('--train_interval', required=False, default=50, type=int)
parser.add_argument('--start_train', required=False, default=10000, type=int)
parser.add_argument('--fill_buffer', required=False, default=20000, type=int)
parser.add_argument('--h_scale', required=False, default=1.0, type=float)
parser.add_argument('--ep_len', required=False, default=None, type=float)
parser.add_argument('--batch_size', required=False, default=128, type=int)
parser.add_argument('--gamma', required=False, default=0.99, type=float)
parser.add_argument('--smooth', action='store_true')
parser.add_argument('--no_double', action='store_false')


args = parser.parse_args()

if args.algo == 'hjdqn':
    for _ in range(args.num_trial):
        run_hjdqn(args.env,
                  L=args.L,
                  gamma=args.gamma,
                  lr=args.lr,
                  sigma=args.sigma,
                  polyak=args.tau,
                  hidden1=args.hidden,
                  hidden2=args.hidden,
                  max_iter=args.max_iter,
                  buffer_size=1e6,
                  fill_buffer=args.fill_buffer,
                  train_interval=args.train_interval,
                  start_train=args.start_train,
                  eval_interval=args.eval_interval,
                  h_scale=args.h_scale,
                  device=args.device,
                  double=args.no_double,
                  smooth=args.smooth,
                  ep_len=args.ep_len,
                  noise=args.noise,
                  batch_size=args.batch_size,
                  render=args.render
                  )

elif args.algo == 'ddpg':
    for _ in range(args.num_trial):
        run_ddpg(args.env,
                 gamma=args.gamma,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 polyak=1e-3,
                 sigma=0.1,
                 hidden_size1=args.hidden,
                 hidden_size2=args.hidden,
                 max_iter=args.max_iter,
                 eval_interval=args.eval_interval,
                 start_train=args.start_train,
                 train_interval=args.train_interval,
                 buffer_size=1e6,
                 fill_buffer=args.fill_buffer,
                 h_scale=args.h_scale,
                 device=args.device,
                 render=args.render
                 )
