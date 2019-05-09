#======================================================
# codes from the fallowing github repository are used in some parts of implementations
#https://github.com/meetshah1995/pytorch-semseg
#======================================================

import torch
import argparse
from torch.autograd import Variable
from torch.utils import data
#from tqdm import tqdm
from src.models import get_model
from src.loader import get_loader, get_data_path
from src.metrics import runningScore
#from src.loss_grad_update import *
from src.augmentations import *
from src.loader.cityscapes_loader import decode_segmap_mine
import matplotlib.pyplot as plt
from matplotlib import cm
plt.switch_backend('agg')
from src.utils_03 import *
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")

def train(args):
    # Setup Dataloader ============================================================
    data_loader_seg = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    v_loader = data_loader_seg(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols), \
                               img_norm=args.img_norm)
    n_classes = v_loader.n_classes
    valloader = data.DataLoader(v_loader, batch_size=1, num_workers=8)   \
        # IN VALIDATION WE CALCULATE ALL OF THE IMAGES ONE BY ONE => MADE BATCH SIZE 1 PREVIOUSLY IT WAS BATCHSIZE
    # Setup Metrics =================================
    running_metrics = runningScore(n_classes)
    # Setup Model=======================================================================================================
    model_features    = get_model('resnet', n_classes)
    model_segment     = get_model('segment',n_classes)
    model_reconstruct = get_model('reconstruct', n_classes=3)
    model_depth       = get_model('depth',n_classes=1)
    #move model to cuda
    model_features = torch.nn.DataParallel(model_features, device_ids=range(torch.cuda.device_count()))
    model_features.cuda()
    model_segment= torch.nn.DataParallel(model_segment, device_ids=range(torch.cuda.device_count()))
    model_segment.cuda()
    model_reconstruct= torch.nn.DataParallel(model_reconstruct, device_ids=range(torch.cuda.device_count()))
    model_reconstruct.cuda()
    model_depth = torch.nn.DataParallel(model_depth, device_ids=range(torch.cuda.device_count()))
    model_depth.cuda()
    #Initializer.initialize(model=model_segment, initialization=init.xavier_uniform)
    #print the model outputs
    summary(model_features,(3, args.img_rows, args.img_cols))
    summary(model_segment, (512, 8, 16))
    #summary(model_reconstruct, (512, 8, 16))
    #summary(model_classify, (512, 8, 16))

    # Load the pretrained =============================================================================================
    if args.resume_feature:
        print("Loading model and optimizer from checkpoint '{}'".format(args.resume_feature))
        # FEATURE NETWORK
        checkpoint_feature = torch.load(args.resume_feature)
        model_features.load_state_dict(checkpoint_feature['model_state'])
        print("Loaded checkpoint '{}' (epoch {})"
                .format(args.resume_feature, checkpoint_feature['epoch']))
    if args.resume_segment:
        print("Loading model and optimizer from checkpoint '{}'".format(args.resume_segment))
        # SEGMENTATION NETWORK
        checkpoint_segment = torch.load(args.resume_segment)
        model_segment.load_state_dict(checkpoint_segment['model_state'],strict=False)
        print("Loaded checkpoint '{}' (epoch {})"
                .format(args.resume_segment, checkpoint_segment['epoch']))
    if args.resume_reconstruct:
        print("Loading model and optimizer from checkpoint '{}'".format(args.resume_reconstruct))
        # RECONSTRUCTION NETWORK
        checkpoint_reconstruct = torch.load(args.resume_reconstruct)
        model_reconstruct.load_state_dict(checkpoint_reconstruct['model_state'],strict=False)
        print("Loaded checkpoint '{}' (epoch {})"
                .format(args.resume_reconstruct, checkpoint_reconstruct['epoch']))
    if args.resume_depth:
        # DEPTH NETWORK
        checkpoint_depth = torch.load(args.resume_depth)
        model_depth.load_state_dict(checkpoint_depth['model_state'],strict=False)
        print("Loaded checkpoint '{}' (epoch {})"
                .format(args.resume_depth, checkpoint_depth['epoch']))

    png_jpg_flag = args.quality #0 means PNG, 95, 90, 85, 80 mean the JPG quantization
    # START Testing ===================================================================================================
    model_features.eval()
    model_segment.eval()
    model_reconstruct.eval()
    model_depth.eval()
   
    for i_val, (images_val, labels_val,obj_label_val, images_val_not_normalized,depth_val) in (enumerate(valloader)): #tqdm
        images_val = Variable(images_val.cuda(), volatile=True)
        images_val_not_normalized = Variable(images_val_not_normalized.cuda(), volatile=True)
        labels_val = Variable(labels_val.cuda(), volatile=True)
        depth_val = Variable(depth_val.cuda(), volatile=True)

        #feed forward -> features
        features_val = model_features(images_val)
        # Quantization (returns the integer features in the range of 0-255)
        features_val,features_val_255, max_to_encode, min_to_encode = Quantize_center_VALIDATION(features_val)
        features_val_255_numpy = features_val_255.data.cpu().numpy()

        #Dumping
        featue_file_name = './dumped_features/temp/val_comp'+ str(i_val)  # +'.bin'
        features_val = dump_feature_2D (features_val_255_numpy,featue_file_name,max_to_encode, min_to_encode,png_jpg_flag)
        #==========================================================

        # feed forward -> Tasks
        outputs_segment_val = model_segment(features_val)
        outputs_reconstruct_val = model_reconstruct(features_val)
        outputs_depth_val = model_depth(features_val)

        gt_label = labels_val.data.cpu().numpy()
        pred_label = outputs_segment_val.data.max(1)[1].cpu().numpy()

        depth_val_gt = depth_val.data.cpu().numpy()
        depth_val_pred = outputs_depth_val.data.cpu().numpy()
        depth_val_pred = depth_val_pred[0, :, :, :]

        # Visualize
        if args.visualize:
            #depth_val_pred   = depth_val_pred[0,:,:,:]
            depth_val_pred_plt   = depth_val_pred.transpose((1, 2, 0))
            depth_val_pred_plt   = depth_val_pred_plt[:,:,0] #np.squeeze(depth_val_pred, axis=2)
            depth_val_gt_plt = depth_val_gt.transpose((1, 2, 0))
            depth_val_gt_plt = depth_val_gt_plt[:, :, 0]

            images_debug = images_val_not_normalized.data.cpu().numpy()
            images_debug = images_debug[0,:,:,:]
            images_debug = images_debug.transpose((1, 2, 0))
            outputs_reconstruct_val_clamp = torch.clamp(outputs_reconstruct_val, 0.0, 1.0)
            recons_debug = outputs_reconstruct_val_clamp.data.cpu().numpy()
            recons_debug = recons_debug[0, :, :, :]
            recons_debug = recons_debug.transpose((1, 2, 0))

            seg_pred_color= decode_segmap_mine(pred_label[0,:,:])
            seg_gt_color = decode_segmap_mine(gt_label[0,:,:])
            f2, axar = plt.subplots(1, 6)
            axar[0].imshow((images_debug*255).astype(np.uint8))
            axar[0].axis('off')
            axar[1].imshow(seg_gt_color)
            axar[1].axis('off')
            axar[2].imshow(((depth_val_gt_plt).astype(np.int32)),cmap=cm.coolwarm)
            axar[2].axis('off')
            axar[3].imshow(recons_debug)
            axar[3].axis('off')
            axar[4].imshow(seg_pred_color)
            axar[4].axis('off')
            axar[5].imshow(((depth_val_pred_plt).astype(np.int32)),cmap=cm.coolwarm)
            axar[5].axis('off')
            f2.savefig('./dumped_images/val_figure'+str(i_val)+'.png',dpi=300)
            #plt.show()

        running_metrics.update(gt_label, pred_label,\
                                   images_val_not_normalized , outputs_reconstruct_val,
                                depth_val_gt,depth_val_pred)
    score, class_iou = running_metrics.get_scores()
    for k, v in score.items():
        print(k, v)
    running_metrics.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--exp', nargs='?', type=str, default='EXP', 
                        help='Experiment name for log file')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=512,
                        help='Height of the input image')
    parser.add_argument('--quality', nargs='?', type=int, default=0,
                        help='Quality of the image codec, 0 means PNG')
    parser.add_argument('--img_cols', nargs='?', type=int, default=512,
                        help='Width of the input image')
    parser.add_argument('--num_obj', nargs='?', type=int, default=10,
                        help='Number of objects in the dataset')
    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--resume_feature', nargs='?', type=str, default=None,
                        help='Path to previous saved feature model to restart from')
    parser.add_argument('--resume_segment', nargs='?', type=str, default=None,
                        help='Path to previous saved segment model to restart from')
    parser.add_argument('--resume_reconstruct', nargs='?', type=str, default=None,
                        help='Path to previous saved reconstruct model to restart from')
    parser.add_argument('--resume_depth', nargs='?', type=str, default=None,
                        help='Path to previous saved depth model to restart from')
    parser.add_argument('--visualize', dest='visualize', action='store_true',
                        help='Enable visualization for network outputs| False by default')
    parser.set_defaults(visualize=False)

    args = parser.parse_args()
    train(args)

