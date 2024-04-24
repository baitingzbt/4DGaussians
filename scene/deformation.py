import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, grid_pe=0, args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires, W)
        self.blend_time_force = args.blend_time_force
        self.args = args
        self.ratio=0        
        self.posbase_pe = args.posebase_pe  # num repeat for points
        self.scale_rotation_pe = args.scale_rotation_pe  # num repeat for rotation
        self.opacity_pe = args.opacity_pe  # num repeat for opacity
        self.grid_pe = args.grid_pe

        self.extra_point_dim = args.posebase_pe * 2 * 3
        self.extra_scale_dim = args.scale_rotation_pe * 2 * 3
        self.extra_rotation_dim = args.scale_rotation_pe * 2 * 4
        self.extra_opacity_dim = args.opacity_pe * 2 * 3
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
        grid_out_dim = self.grid.feat_dim + self.grid_pe * (self.grid.feat_dim) * 2 

        self.feature_out = [nn.Linear(grid_out_dim, self.W, dtype=torch.float32, bias=True)]
        for i in range(self.D - 1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W, dtype=torch.float32, bias=True))
        self.feature_out = nn.Sequential(*self.feature_out)

        self.feature_out2 = nn.Linear(256 + 64, 256)

        self.pos_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W + self.extra_point_dim, self.W, dtype=torch.float32, bias=True),
            nn.ReLU(), nn.Linear(self.W, 3, dtype=torch.float32, bias=True)
        )
        self.scales_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W + self.extra_scale_dim, self.W, dtype=torch.float32, bias=True),
            nn.ReLU(), nn.Linear(self.W, 3, dtype=torch.float32, bias=True)
        )
        self.rotations_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W + self.extra_rotation_dim, self.W, dtype=torch.float32, bias=True),
            nn.ReLU(), nn.Linear(self.W, 4, dtype=torch.float32, bias=True)
        )
        self.opacity_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W + self.extra_opacity_dim, self.W, dtype=torch.float32, bias=True),
            nn.ReLU(), nn.Linear(self.W, 1, dtype=torch.float32, bias=True)
        )
        self.force_embedder = nn.Sequential(
            nn.Linear(2, 1)
        )

        if self.blend_time_force:
            # self.force_time_embed = nn.Sequential(nn.ReLU(), nn.Linear(2, 1))
            self.force_time_embed = nn.Sequential(nn.Linear(2, 1))

        self.shs_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W, dtype=torch.float32, bias=True),
            nn.ReLU(), nn.Linear(self.W, 16 * 3, dtype=torch.float32, bias=True)
        )
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
            nn.Linear(128, 16),
        )
        self.frames_interact = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def query_time_force(self, rays_pts_emb, time_emb, force_emb, hidden, prev_frames_emb):
        ''' embedding force from dim4 to dim1, to reduce numerical instability '''
        if force_emb is not None:  # NOTE: if use_force = True
            force_emb = self.force_embedder(force_emb)

        if self.blend_time_force and force_emb is not None:
            time_emb = self.force_time_embed(torch.cat((time_emb, force_emb), dim=1))
            force_emb = None  # NOTE: force information merged into time
        
        
        # force_emb is None if blending_time_force
        grid_feature = self.grid.forward(rays_pts_emb[:, :3], time_emb, force_emb, hidden) # [n_pts, xxx]
        if self.grid_pe > 1:
            grid_feature = poc_fre(grid_feature, self.grid_pe)
            grid_feature = torch.cat([grid_feature], -1).to(dtype=torch.float32)
        hidden2 = self.feature_out2(torch.cat((grid_feature, prev_frames_emb), dim=1))
        hidden3 = self.feature_out(hidden2)
        return hidden3

    @property
    def get_empty_ratio(self):
        return self.ratio

    def forward_dynamic(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_emb, force_emb, hidden_emb, prev_frames):
        time_input = time_emb[:, :1] if self.args.use_time else None
        force_input = force_emb if self.args.use_force else None
        # use more poc_fre embedding?
        # 1. completely replace hexplane by MLP
        # 2. pre-process and downsize embedding back to its dimension then feed into hexplane
        # 3. use embedding only in deform, don't use it for hexplane
        n_pts = rays_pts_emb.shape[0]
        prev_frames_emb = self.prev_frame_encoder(prev_frames.cuda())
        prev_frames_emb = self.frames_interact(prev_frames_emb.flatten().repeat(n_pts, 1))
        hidden = self.query_time_force(rays_pts_emb, time_input, force_input, hidden_emb, prev_frames_emb)
        mask = torch.ones_like(opacity_emb[:, 0], dtype=torch.float32).unsqueeze(-1)
        if self.args.no_dx:
            pts = rays_pts_emb[:, :3]
        else:
            # DEFAULT
            pts = rays_pts_emb[:, :3] * mask + self.pos_deform(torch.cat((hidden, rays_pts_emb[:, 3:]), dim=1))

        if self.args.no_ds:
            scales = scales_emb[:, :3]
        else:
            # DEFAULT
            # scales = scales_emb[:, :3] * mask + self.scales_deform(hidden)
            scales = scales_emb[:, :3] * mask + self.scales_deform(torch.cat((hidden, scales_emb[:, 3:]), dim=1))

        if self.args.no_dr:
            rotations = rotations_emb[:, :4]
        else:
            # DEFAULT
            # dr = self.rotations_deform(hidden)
            dr = self.rotations_deform(torch.cat((hidden, rotations_emb[:, 4:]), dim=1))
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
            opacity = opacity_emb[:, :1] * mask + self.opacity_deform(hidden)

        if self.args.no_dshs:
            shs = shs_emb
        else:
            # DEFAULT
            shs = shs_emb * mask.unsqueeze(-1) + self.shs_deform(hidden).reshape([shs_emb.shape[0], 16, 3])

        return pts, scales, rotations, opacity, shs, hidden
    
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
        self.apply(initialize_weights)
        self.recur = False
        self.recur_state = False
        self.recur_hidden = False

    @property
    def get_aabb(self):
        return self.deformation_net.get_aabb
    
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
    # Gaussian -> DeformationNet -> HexPlane
    # xyz_init -> poc_fre -> HexPlane -> HiddenFeature -> deform
    def forward_dynamic(self, points_in, scales_in, rotations_in, opacity_in, shs_in, times_sel_in, force_in, hidden_in, prev_frames):
        point_emb = poc_fre(points_in, self.pos_poc)
        scales_emb = poc_fre(scales_in, self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations_in, self.rotation_scaling_poc)
        if hidden_in is None and self.recur:
            hidden_in = torch.zeros((point_emb.shape[0], 64), dtype=torch.float32, device='cuda')
        means3D, scales, rotations, opacity, shs, hidden = self.deformation_net.forward_dynamic(
            point_emb, scales_emb, rotations_emb, opacity_in, shs_in, times_sel_in, force_in, hidden_in, prev_frames
        )
        return means3D, scales, rotations, opacity, shs, hidden
    
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() # + list(self.timenet.parameters())

    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

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