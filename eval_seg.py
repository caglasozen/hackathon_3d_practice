import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")


    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default="seg", help='The task: cls or seg')
    parser.add_argument('--batch_size', type=int, default=4, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')


    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(k=args.num_seg_class)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    # ------ TO DO: Make Prediction ------
    test_dataloader = get_data_loader(args=args, train=False)

    correct_point = 0
    num_point = 0
    
    model = model.to(args.device)
    
    for batch in test_dataloader:
        point_clouds, labels = batch
        point_clouds = point_clouds.transpose(1, 2)
        point_clouds = point_clouds.to(args.device)
        labels = labels.to(args.device).to(torch.long)

        # ------ TO DO: Make Predictions ------
        with torch.no_grad():
            pred, trans, trans_feat = model(point_clouds)
            
        pred_labels = pred.data.max(2)[1]
        correct_point += pred_labels.eq(labels.data).cpu().sum().item()
        num_point += labels.view([-1,1]).size()[0]

    # Compute Accuracy of Test Dataset
    accuracy = correct_point / num_point

    print ("test accuracy: {}".format(accuracy))
    
    # Visualize Segmentation Result (Pred VS Ground Truth)
    viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
    viz_seg(test_data[args.i], pred_labels[args.i], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)
