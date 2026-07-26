import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualizer.droid_visualizer import visualization_fn
            self.visualizer = Process(target=visualization_fn, args=(self.video, None))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)


    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from modules.extractor import BasicEncoder
from modules.corr import CorrBlock
from modules.gru import ConvGRU
from modules.clipping import GradientClip

from lietorch import SE3
from geom.ba import BA

import geom.projective_ops as pops
from geom.graph_utils import graph_to_edge_list, keyframe_indicies

from torch_scatter import scatter_mean


def cvx_upsample(data, mask):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3,3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)

    return up_data

def upsample_disp(disp, mask):
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch*num, ht, wd, 1)
    mask = mask.view(batch*num, -1, ht, wd)
    return cvx_upsample(disp, mask).view(batch, num, 8*ht, 8*wd)


class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip(),
            nn.Softplus())

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8*8*9, 1, padding=0))

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))

        net = net.view(batch, num, 128, ht, wd)
        net = scatter_mean(net, ix, dim=1)
        net = net.view(-1, 128, ht, wd)

        net = self.relu(self.conv2(net))

        eta = self.eta(net).view(batch, -1, ht, wd)
        upmask = self.upmask(net).view(batch, -1, 8*8*9, ht, wd)

        return .01 * eta, upmask


class UpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2*3 + 1)**2

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True))

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip(),
            nn.Sigmoid())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip())

        self.gru = ConvGRU(128, 128+128+64)
        self.agg = GraphAgg()

    def forward(self, net, inp, corr, flow=None, ii=None, jj=None):
        """ RaftSLAM update operator """

        batch, num, ch, ht, wd = net.shape

        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)

        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch*num, -1, ht, wd)
        inp = inp.view(batch*num, -1, ht, wd)        
        corr = corr.view(batch*num, -1, ht, wd)
        flow = flow.view(batch*num, -1, ht, wd)

        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)

        ### update variables ###
        delta = self.delta(net).view(*output_dim)
        weight = self.weight(net).view(*output_dim)

        delta = delta.permute(0,1,3,4,2)[...,:2].contiguous()
        weight = weight.permute(0,1,3,4,2)[...,:2].contiguous()

        net = net.view(*output_dim)

        if ii is not None:
            eta, upmask = self.agg(net, ii.to(net.device))
            return net, delta, weight, eta, upmask

        else:
            return net, delta, weight


class DroidNet(nn.Module):
    def __init__(self):
        super(DroidNet, self).__init__()
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')
        self.update = UpdateModule()


    def extract_features(self, images):
        """ run feeature extraction networks """

        # normalize images
        images = images[:, :, [2,1,0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        fmaps = self.fnet(images)
        net = self.cnet(images)
        
        net, inp = net.split([128,128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return fmaps, net, inp


    def forward(self, Gs, images, disps, intrinsics, graph=None, num_steps=12, fixedp=2):
        """ Estimates SE3 or Sim3 between pair of frames """

        u = keyframe_indicies(graph)
        ii, jj, kk = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        fmaps, net, inp = self.extract_features(images)
        net, inp = net[:,ii], inp[:,ii]
        corr_fn = CorrBlock(fmaps[:,ii], fmaps[:,jj], num_levels=4, radius=3)

        ht, wd = images.shape[-2:]
        coords0 = pops.coords_grid(ht//8, wd//8, device=images.device)
        
        coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
        target = coords1.clone()

        Gs_list, disp_list, residual_list = [], [], []
        for step in range(num_steps):
            Gs = Gs.detach()
            disps = disps.detach()
            coords1 = coords1.detach()
            target = target.detach()

            # extract motion features
            corr = corr_fn(coords1)
            resd = target - coords1
            flow = coords1 - coords0

            motion = torch.cat([flow, resd], dim=-1)
            motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            net, delta, weight, eta, upmask = \
                self.update(net, inp, corr, motion, ii, jj)

            target = coords1 + delta

            for i in range(2):
                Gs, disps = BA(target, weight, eta, Gs, disps, intrinsics, ii, jj, fixedp=2)

            coords1, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
            residual = (target - coords1)

            Gs_list.append(Gs)
            disp_list.append(upsample_disp(disps, upmask))
            residual_list.append(valid_mask * residual)


        return Gs_list, disp_list, residual_list
import torch
import lietorch
import numpy as np

import time
from lietorch import SE3
from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidAsyncBackend
from trajectory_filler import PoseTrajectoryFiller
from align import align_pose_fragements

from collections import OrderedDict
from torch.multiprocessing import Process


def load_network(weights, device="cuda:0"):
    net = DroidNet()
    state_dict = OrderedDict(
        [(k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()]
    )

    state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
    state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
    state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
    state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

    net.load_state_dict(state_dict)
    net.to(device=device)
    net.eval()

    return net


def backend_process(args, depth_video1, depth_video2, device="cuda"):
    torch.set_num_threads(8)

    seperate_device = False
    if device != "cuda":
        seperate_device = True
        torch.cuda.set_device(device)

    with torch.no_grad():
        net = load_network(args.weights, device=device)

        # use more compute if running backend on seperate device
        sleep_time = 10
        num_iters = 12 if seperate_device else 8

        backend = DroidAsyncBackend(net, depth_video2, args)

        while 1:
            if depth_video1.counter.value > 32 or depth_video2.ready.value:
                # check if the end of the video has been reached
                is_last_iteration = depth_video2.ready.value

                # don't align scale for RGB-D or stereo videos
                align_scale = not depth_video2.stereo and not torch.any(
                    depth_video1.disps_sens
                )

                if is_last_iteration:
                    t0 = max(depth_video2.counter.value - 2, 0)
                    t1 = depth_video1.counter.value

                else:
                    t0 = max(depth_video2.counter.value - 2, 0)
                    # avoid keyframing area
                    t1 = depth_video1.counter.value - 5

                with depth_video1.get_lock():
                    pose1_copy = depth_video1.poses.clone().to(device=device)
                    disps1_copy = depth_video1.disps.clone().to(device=device)

                if t0 > 0:
                    # align pose and scale
                    dP, s = align_pose_fragements(
                        pose1_copy[t0 - 10 : t0 - 1],
                        depth_video2.poses[t0 - 10 : t0 - 1],
                    )

                    if not align_scale:
                        s = 1.0

                    pose1_copy.data[..., :3] *= s

                else:
                    s = 1.0
                    dP = SE3.IdentityLike(SE3(depth_video2.poses[[t0]]))

                with depth_video1.get_lock():
                    depth_video2.poses[t0:t1] = (dP * SE3(pose1_copy[t0:t1])).data.to(
                        device=device
                    )
                    depth_video2.disps[t0:t1] = disps1_copy[t0:t1].to(device=device) / s

                    depth_video2.disps_sens[t0:t1] = depth_video1.disps_sens[t0:t1].to(
                        device=device
                    )
                    depth_video2.images[t0:t1] = depth_video1.images[t0:t1].to(
                        device=device
                    )
                    depth_video2.tstamp[t0:t1] = depth_video1.tstamp[t0:t1].to(
                        device=device
                    )
                    depth_video2.intrinsics[t0:t1] = depth_video1.intrinsics[t0:t1].to(
                        device=device
                    )
                    depth_video2.fmaps[t0:t1] = depth_video1.fmaps[t0:t1].to(
                        device=device
                    )
                    depth_video2.nets[t0:t1] = depth_video1.nets[t0:t1].to(
                        device=device
                    )
                    depth_video2.inps[t0:t1] = depth_video1.inps[t0:t1].to(
                        device=device
                    )

                depth_video2.counter.value = t1
                backend(num_iters, normalize=False)

                if is_last_iteration:
                    break

                if not depth_video2.ready.value > 0:
                    time.sleep(sleep_time)

    del depth_video2


class DroidAsync:
    def __init__(self, args):
        super(DroidAsync, self).__init__()
        net = load_network(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        torch.set_num_threads(8)

        self.frontend_device = (
            "cuda" if not hasattr(args, "frontend_device") else args.frontend_device
        )
        self.backend_device = (
            "cuda" if not hasattr(args, "backend_device") else args.backend_device
        )

        net = load_network(args.weights, device=self.frontend_device)

        self.video1 = DepthVideo(
            args.image_size,
            args.buffer,
            stereo=args.stereo,
            device=self.frontend_device,
        )
        self.video2 = DepthVideo(
            args.image_size, args.buffer, stereo=args.stereo, device=self.backend_device
        )

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(net, self.video1, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(net, self.video1, self.args)

        # backend process
        self.backend_proc = Process(
            target=backend_process,
            args=(args, self.video1, self.video2, self.backend_device),
        )
        self.backend_proc.start()

        # visualizer
        if not self.disable_vis:
            from visualizer.droid_visualizer import visualization_fn

            self.visualizer = Process(
                target=visualization_fn,
                args=(
                    self.video1,
                    self.video2,
                ),
            )
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(net, self.video2)

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """main thread - update map"""

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

    def terminate(self, stream=None):
        """terminate the visualization process, return poses [t, q]"""

        self.video2.ready.value = 1
        self.backend_proc.join()

        del self.frontend
        del self.video1

        torch.cuda.empty_cache()

        # fill in missing non-keyframe poses
        self.traj_filler.video = self.traj_filler.video.to(self.frontend_device)
        camera_trajectory = self.traj_filler(stream)

        return camera_trajectory.inv().data.cpu().numpy()
