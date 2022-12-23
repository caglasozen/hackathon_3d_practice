import torch
from train import create_parser
from eval_cls import create_parser, evaluate

parser = create_parser()
args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
args.checkpoint_dir = args.checkpoint_dir+"/"+args.task # checkpoint directory is task specific

#Robustness analysis with no_of_points 

#Classification

exp_vals = [10, 100, 1000, 5000, 10000]

for no_of_points in exp_vals:
    args.num_points = no_of_points
    
    print("##############EXPERIMENT FOR: " + no_of_points )
    evaluate(args)
