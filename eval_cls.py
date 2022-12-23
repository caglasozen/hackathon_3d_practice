import numpy as np
import argparse

import torch
from models import cls_model
from data_loader import get_data_loader
from utils import create_dir, viz_cls


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--batch_size', type=int, default=4, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


def evaluate(args):
    # ------ TO DO: Initialize Model for Classification Task ------
    model =  cls_model(k=3)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    test_dataloader = get_data_loader(args=args, train=False)

    correct_obj = 0
    num_obj = 0
    
    model = model.to(args.device)
    
    for i, batch in enumerate(test_dataloader):
        point_clouds, labels = batch
        point_clouds = point_clouds.transpose(1, 2)
        point_clouds = point_clouds.to(args.device)
        labels = labels.to(args.device).to(torch.long)

        # ------ TO DO: Make Predictions ------
        with torch.no_grad():
            pred, trans = model(point_clouds)
            
        pred_labels = pred.data.max(1)[1]
        correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
        num_obj += labels.size()[0]
        
        if i < 50:
            viz_cls(point_clouds, labels, pred_labels, args.output_dir, i)

    # Compute Accuracy of Test Dataset
    accuracy = correct_obj / num_obj

    print ("test accuracy: {}".format(accuracy))


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    #Robustness analysis with no_of_points 

    #Classification

    exp_vals = [10, 100, 1000, 5000, 10000]

    for no_of_points in exp_vals:
        args.num_points = no_of_points
        
        print("##############EXPERIMENT FOR: " + str(no_of_points) )
        evaluate(args)
