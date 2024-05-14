import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid

def init_zero_(model: nn.Sequential):
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.weight, 0)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)  # Set biases to zero

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, grid_pe=0, args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.grid_pe = grid_pe
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires, W)
        self.blend_time_force = args.blend_time_force
        self.args = args
        self.ratio=0        
        self.posbase_pe = args.posebase_pe  # num repeat for points
        self.scale_rotation_pe = args.scale_rotation_pe  # num repeat for rotation
        self.opacity_pe = args.opacity_pe  # num repeat for opacity
        self.grid_pe = args.grid_pe
        self.force_pe = args.force_pe
        self.time_pe = args.time_pe
        self.extra_point_dim = args.posebase_pe * 2 * 3
        self.extra_scale_dim = args.scale_rotation_pe * 2 * 3
        self.extra_rotation_dim = args.scale_rotation_pe * 2 * 4
        self.extra_opacity_dim = args.opacity_pe * 2 * 3
        self.prev_frames = args.prev_frames
        self.prev_frames_out_dim = 16
        # full_dim = dim + num_repeat * 2 * dim    (e.g. dim=3 for points/opacity, dim=4 for rotations)
        self.create_net()

    @property
    def get_aabb(self):
        return self.grid.get_aabb

    def set_aabb(self, xyz_max, xyz_min):
        # print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    
    def create_net(self):
        # expand dimension if self.grid_pe != 0
        self.prev_frame_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=4),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, self.prev_frames_out_dim),
        )
        total_prev_frames_dim = self.prev_frames * self.prev_frames_out_dim
        grid_out_dim = self.grid.feat_dim + self.grid_pe * (self.grid.feat_dim) * 2 
        self.feature_out1 = nn.Linear(grid_out_dim + total_prev_frames_dim, grid_out_dim)
        self.feature_out2 = [nn.Linear(grid_out_dim, self.W, dtype=torch.float32, bias=True)]
        for i in range(self.D - 1):
            self.feature_out2.append(nn.ReLU())
            self.feature_out2.append(nn.Linear(self.W, self.W, dtype=torch.float32, bias=True))
        self.feature_out2 = nn.Sequential(*self.feature_out2)
        deform_extra_dim = 1 + 2 * self.force_pe + 1 + 2 * self.time_pe
        self.pos_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W + self.extra_point_dim + deform_extra_dim, self.W, dtype=torch.float32, bias=True),
            # nn.ReLU(), nn.Linear(self.W, self.W),
            # nn.ReLU(), nn.Linear(self.W, self.W),
            # nn.ReLU(), nn.Linear(self.W, self.W),
            nn.ReLU(), nn.Linear(self.W, 3, dtype=torch.float32, bias=True)
        )
        self.scales_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W + self.extra_scale_dim + deform_extra_dim, self.W, dtype=torch.float32, bias=True),
            # nn.ReLU(), nn.Linear(self.W + self.extra_scale_dim, self.W, dtype=torch.float32, bias=True),
            # nn.ReLU(), nn.Linear(self.W, self.W),
            # nn.ReLU(), nn.Linear(self.W, self.W),
            nn.ReLU(), nn.Linear(self.W, 3, dtype=torch.float32, bias=True)
        )
        init_zero_(self.scales_deform)
        self.rotations_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W + self.extra_rotation_dim + deform_extra_dim, self.W, dtype=torch.float32, bias=True),
            # nn.ReLU(), nn.Linear(self.W, self.W),
            # nn.ReLU(), nn.Linear(self.W, self.W),
            nn.ReLU(), nn.Linear(self.W, 4, dtype=torch.float32, bias=True)
        )
        init_zero_(self.rotations_deform)
        self.opacity_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W + self.extra_opacity_dim + deform_extra_dim, self.W, dtype=torch.float32, bias=True),
            # nn.ReLU(), nn.Linear(self.W, self.W),
            # nn.ReLU(), nn.Linear(self.W, self.W),
            nn.ReLU(), nn.Linear(self.W, 1, dtype=torch.float32, bias=True)
        )
        init_zero_(self.opacity_deform)
        self.shs_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W + deform_extra_dim, self.W, dtype=torch.float32, bias=True),
            # nn.ReLU(), nn.Linear(self.W, self.W),
            nn.ReLU(), nn.Linear(self.W, 16 * 3, dtype=torch.float32, bias=True)
        )
        init_zero_(self.shs_deform)

        # self.force_embedder = nn.Linear(21, 1)
        self.force_embedder = nn.Linear(2, 1)
        if self.blend_time_force:
            # self.force_time_embed = nn.Sequential(nn.ReLU(), nn.Linear(2, 1))
            self.force_time_embed = nn.Linear(2, 1)

    def query_time_force(self, rays_pts_emb, time_emb, force_emb, prev_frames_emb):
        # NOTE: if use_force = True, compress if force to dim1 if not already
        # if force_emb is not None and force_emb.shape[1] != 1:
        # force_emb = self.force_embedder(force_emb[:, :1])
        if self.blend_time_force:
            time_emb = self.force_time_embed(torch.cat((time_emb[:, :1], force_emb[:, :1]), dim=1))
            force_emb = None  # NOTE: force information merged into time
        grid_feature = self.grid.forward(rays_pts_emb[:, :3], time_emb, force_emb) # [n_pts, xxx]
        # if self.grid_pe > 1:
        #     grid_feature = poc_fre(grid_feature, self.grid_pe)
        #     grid_feature = torch.cat([grid_feature], -1).to(dtype=torch.float32)

        ### tmp hack below ###
        # time_emb = self.force_time_embed(torch.cat((time_emb, self.force_embedder(force_emb)), dim=1))
        # grid_feature = self.grid.forward(rays_pts_emb[:, :3], time_emb, None) # [n_pts, xxx]
        ### tmp hack above ###
        if prev_frames_emb is not None:
            grid_feature = self.feature_out1(torch.cat((grid_feature, prev_frames_emb), dim=1))
        return self.feature_out2(grid_feature)

    @property
    def get_empty_ratio(self):
        return self.ratio

    def forward_dynamic(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_emb, force_emb, prev_frames):
        time_emb = time_emb if self.args.use_time else time_emb * 0.0
        force_emb = force_emb if self.args.use_force else force_emb * 0.0
        # use more poc_fre embedding?
        # 1. completely replace hexplane by MLP
        # 2. pre-process and downsize embedding back to its dimension then feed into hexplane
        # 3. use embedding only in deform, don't use it for hexplane
        n_pts = rays_pts_emb.shape[0]
        prev_frames_emb = None if prev_frames is None else self.prev_frame_encoder(prev_frames.cuda()).flatten().repeat(n_pts, 1)
        hidden = self.query_time_force(rays_pts_emb, time_emb, force_emb, prev_frames_emb)
        if self.args.no_dx:
            pts = rays_pts_emb[:, :3]
        else:
            # DEFAULT
            # fastest = torch.topk(dx[:, :2].abs().sum(dim=1), k=1500, largest=True, dim=0).indices.detach().cpu() # xy only, drop z
            pts = rays_pts_emb[:, :3] + self.pos_deform(
                torch.cat((hidden, rays_pts_emb[:, 3:], time_emb, force_emb), dim=1))

        if self.args.no_ds:
            scales = scales_emb[:, :3]
        else:
            # DEFAULT
            # scales = scales_emb[:, :3] * mask + self.scales_deform(torch.cat((hidden, scales_emb[:, 3:]), dim=1))
            scales = scales_emb[:, :3] + self.scales_deform(
                torch.cat((hidden, scales_emb[:, 3:], time_emb, force_emb), dim=1))

        if self.args.no_dr:
            rotations = rotations_emb[:, :4]
        else:
            # DEFAULT
            # dr = self.rotations_deform(hidden)
            dr = self.rotations_deform(torch.cat((hidden, rotations_emb[:, 4:], time_emb, force_emb), dim=1))
            if self.args.apply_rotation:
                # DEFAULT
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:, :4] + dr

        if self.args.no_do:
            # DEFAULT, assume no change in opacity
            opacity = opacity_emb[:, :1] 
        else:
            # opacity = opacity_emb[:, :1] * mask + self.opacity_deform(hidden)
            opacity = opacity_emb[:, :1] + self.opacity_deform(torch.cat((hidden, time_emb, force_emb), dim=1))

        if self.args.no_dshs:
            shs = shs_emb
        else:
            # DEFAULT
            shs = shs_emb + self.shs_deform(
                torch.cat((hidden, time_emb, force_emb), dim=1)).reshape([shs_emb.shape[0], 16, 3])
            # shs = shs_emb * mask.unsqueeze(-1) + self.shs_deform(hidden).reshape([shs_emb.shape[0], 16, 3])

        return pts, scales, rotations, opacity, shs
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
    
    def enable_extra_deform(self):
        self.args.no_ds=False
        self.args.no_dr=False
        self.args.no_dshs=False
    
    def disable_extra_deform(self):
        self.args.no_ds=True
        self.args.no_dr=True
        self.args.no_dshs=True
    

# THE NETWORK THAT OUTPUTS RENDERED IMAGES
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        grid_pe = args.grid_pe
        force_pe = args.force_pe
        time_pe = args.time_pe
        self.deformation_net = Deformation(
            W=net_width,
            D=defor_depth,
            grid_pe=grid_pe,
            args=args
        )
        # self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        # TODO: encoding pocfre on force & time, then use them for deformation functions only (not hexplane)
        # (f, t) -> (xt, yt, zt)    no coarse xyz, so force and time are the only and dominant inputs
        self.register_buffer('force_poc', torch.FloatTensor([(2**i) for i in range(force_pe)]))
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(time_pe)]))
        self.apply(initialize_weights)
        self.recur = False
        self.prev_frames = 0

    @property
    def get_aabb(self):
        return self.deformation_net.get_aabb
    
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
    # Gaussian -> DeformationNet -> HexPlane
    # xyz_init -> poc_fre -> HexPlane -> HiddenFeature -> deform
    def forward_dynamic(self, points_in, scales_in, rotations_in, opacity_in, shs_in, times_sel_in, force_in, prev_frames):
        point_emb = poc_fre(points_in, self.pos_poc)
        scales_emb = poc_fre(scales_in, self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations_in, self.rotation_scaling_poc)
        force_emb = poc_fre(force_in, self.force_poc)
        time_emb = poc_fre(times_sel_in, self.time_poc)
        means3D, scales, rotations, opacity, shs = self.deformation_net.forward_dynamic(
            point_emb, scales_emb, rotations_emb, opacity_in, shs_in, time_emb, force_emb, prev_frames
        )
        return means3D, scales, rotations, opacity, shs # , fastest
        
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() # + list(self.timenet.parameters())

    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()
    
    def forward(self, *args):
        return self.forward_dynamic(*args)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight, gain=1)

def poc_fre(input_data, poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_emb = torch.cat([
        input_data,
        input_data_emb.sin(),
        input_data_emb.cos()
    ], -1)
    return input_data_emb