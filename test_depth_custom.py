import torch
from torch.autograd import Variable
from PIL import Image
from scipy import interpolate
from scipy.ndimage.interpolation import zoom
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
from utils import tensor2array
import models
from cv2 import resize as imresize
from loss_functions_summary import spatial_normalize

parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dispnet", dest='dispnet', type=str, default='DispResNet6', help='dispnet architecture')
parser.add_argument("--posenet", dest='posenet', type=str, default='PoseExpNet', help='posenet architecture')
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--pretrained-posenet", default=None, type=str, help="pretrained PoseNet path (for scale factor)")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--spatial-normalize", action='store_true', help="spatial normalization")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80, type=float)

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")

parser.add_argument("--gt-type", default='KITTI', type=str, help="GroundTruth data type", choices=['npy', 'png', 'KITTI', 'stillbox', 'pfm'])
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")


def main():
    args = parser.parse_args()
    if args.gt_type == 'KITTI':
        from kitti_eval.depth_evaluation_utils import test_framework_KITTI as test_framework
    elif args.gt_type == 'stillbox':
        from stillbox_eval.depth_evaluation_utils import test_framework_stillbox as test_framework
    elif args.gt_type == 'pfm':
        from pfm_eval.depth_evaluation_utils import test_framework_stillbox as test_framework

    disp_net = getattr(models, args.dispnet)().cuda()
    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    if args.pretrained_posenet is None:
        print('no PoseNet specified, scale_factor will be determined by median ratio, which is kiiinda cheating\
            (but consistent with original paper)')
        seq_length = 0
    else:
        weights = torch.load(args.pretrained_posenet)
        seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
        pose_net = getattr(models, args.posenet)(nb_ref_imgs=seq_length - 1, output_exp=False).cuda()
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    dataset_dir = Path(args.dataset_dir)
    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = list(f.read().splitlines())
    else:
        test_files = [file.relpathto(dataset_dir) for file in sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])]

    framework = test_framework(dataset_dir, test_files, seq_length, args.min_depth, args.max_depth)

    print('{} files to test'.format(len(test_files)))
    errors = np.zeros((2, 7, len(test_files)), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        viz_dir = output_dir/'viz'
        print("Saving output to", viz_dir)
        output_dir.makedirs_p()
        viz_dir.makedirs_p()

    for j, sample in enumerate(tqdm(framework)):
        tgt_img = sample['tgt']
        ref_imgs = sample['ref']

        h,w,_ = tgt_img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            tgt_img = imresize(tgt_img, (args.img_height, args.img_width)).astype(np.float32)
            ref_imgs = [imresize(img, (args.img_height, args.img_width)).astype(np.float32) for img in ref_imgs]

        tgt_img = np.transpose(tgt_img, (2, 0, 1))
        ref_imgs = [np.transpose(img, (2, 0, 1)) for img in ref_imgs]

        tgt_img = torch.from_numpy(tgt_img).unsqueeze(0)
        tgt_img = ((tgt_img/255 - 0.5)/0.5).cuda()
        tgt_img_var = Variable(tgt_img, volatile=True)

        ref_imgs_var = []
        for i, img in enumerate(ref_imgs):
            img = torch.from_numpy(img).unsqueeze(0)
            img = ((img/255 - 0.5)/0.5).cuda()
            ref_imgs_var.append(Variable(img, volatile=True))

        pred_disp = disp_net(tgt_img_var)
        if args.spatial_normalize:
            pred_disp = spatial_normalize(pred_disp)
        pred_disp = pred_disp.data.cpu().numpy()[0, 0]

        if args.output_dir is not None:
            if j == 0:
                predictions = np.zeros((len(test_files), *pred_disp.shape))
            predictions[j] = 1/pred_disp

            depth_viz = tensor2array(torch.FloatTensor(pred_disp), max_value=None, colormap='magma')
            depth_viz = np.transpose(depth_viz, (1, 2, 0))
            depth_viz_im = Image.fromarray((255*depth_viz).astype('uint8'))
            depth_viz_im.save(viz_dir/str(j).zfill(4)+'depth.png')

    mean_errors = errors.mean(2)
    error_names = ['abs_rel','sq_rel','rms','log_rms','a1','a2','a3']
    if args.pretrained_posenet:
        print("Results with scale factor determined by PoseNet : ")
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[0]))

    print("Results with scale factor determined by GT/prediction ratio (like the original paper) : ")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[1]))

    if args.output_dir is not None:
        np.save(output_dir/'predictions.npy', predictions)

def interp_gt_disp(mat, mask_val=0):
    mat[mat==mask_val] = np.nan
    x = np.arange(0, mat.shape[1])
    y = np.arange(0, mat.shape[0])
    mat = np.ma.masked_invalid(mat)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~mat.mask]
    y1 = yy[~mat.mask]
    newarr = mat[~mat.mask]

    GD1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear', fill_value=mask_val)
    return GD1

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


if __name__ == '__main__':
    main()
