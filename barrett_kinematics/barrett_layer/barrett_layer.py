import numpy as np
import torch
import math
import trimesh
import glob
import torch
import os

# All lengths are in mm and rotations in radians


class BarrettLayer(torch.nn.Module):
    def __init__(self, device='cpu'):
        # The forward kinematics equations implemented here are from https://support.barrett.com/wiki/Hand/280/KinematicsJointRangesConversionFactors
        super().__init__()
        self.Aw = torch.tensor(0.001 * 25, dtype=torch.float32, device=device)
        self.A1 = torch.tensor(0.001 * 50, dtype=torch.float32, device=device)
        self.A2 = torch.tensor(0.001 * 70, dtype=torch.float32, device=device)
        self.A3 = torch.tensor(0.001 * 50, dtype=torch.float32, device=device)
        self.Dw = torch.tensor(0.001 * 76, dtype=torch.float32, device=device)
        self.Dw_knuckle = torch.tensor(0.001 * 42, dtype=torch.float32, device=device)
        self.D3 = torch.tensor(0.001 * 9.5, dtype=torch.float32, device=device)
        self.phi2 = torch.tensor(0, dtype=torch.float32, device=device)
        self.phi3 = torch.tensor(0 * math.radians(42), dtype=torch.float32, device=device)
        self.pi_0_5 = torch.tensor(math.pi / 2, dtype=torch.float32, device=device)
        self.device = device
        self.meshes = self.load_meshes()
        self.palm = self.meshes["palm_280"][0]
        self.knuckle = self.meshes["knuckle"][0]
        self.finger = self.meshes["finger"][0]
        self.finger_tip = self.meshes["finger_tip"][0]

        self.gripper_faces = [
            self.meshes["palm_280"][1], self.meshes["knuckle"][1], self.meshes["finger"][1],
            self.meshes["finger_tip"][1]
        ]
        self.vertice_face_areas = [
            self.meshes["palm_280"][2], self.meshes["knuckle"][2], self.meshes["finger"][2],
            self.meshes["finger_tip"][2]
        ]
        self.num_vertices_per_part = [
            self.meshes["palm_280"][0].shape[0], self.meshes["knuckle"][0].shape[0], self.meshes["finger"][0].shape[0],
            self.meshes["finger_tip"][0].shape[0]
        ]
        # r and j are used to calculate the forward kinematics for the barrett Hand's different fingers
        self.r = [-1, 1, 0]
        self.j = [1, 1, -1]

    def load_meshes(self):
        mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/../meshes/barrett_hand/*"
        mesh_files = glob.glob(mesh_path)
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]
        meshes = {}
        for mesh_file in mesh_files:
            name = os.path.basename(mesh_file)[:-4]
            mesh = trimesh.load(mesh_file)
            triangle_areas = trimesh.triangles.area(mesh.triangles)
            vert_area_weight = []
            for i in range(mesh.vertices.shape[0]):
                vert_neighour_face = np.where(mesh.faces == i)[0]
                vert_area_weight.append(1000000*triangle_areas[vert_neighour_face].mean())
            temp = torch.ones(mesh.vertices.shape[0], 1).float()
            meshes[name] = [
                torch.cat((torch.FloatTensor(mesh.vertices), temp), dim=-1).to(self.device),
                torch.LongTensor(mesh.faces).to(self.device), torch.FloatTensor(vert_area_weight).to(self.device),
                torch.FloatTensor(mesh.vertex_normals)
            ]
        return meshes

    def forward(self, pose, theta):
        """[summary]
        Args:
            pose (Tensor (batch_size x 4 x 4)): The pose of the base link of the hand as a translation matrix.
            theta (Tensor (batch_size x 7)): The seven degrees of freedome of the Barrett hand. The first column specifies the angle between 
            fingers F1 and F2,  the second to fourth column specifies the joint angle around the proximal link of each finger while the fifth
            to the last column specifies the joint angle around the distal link for each finger

       """
        batch_size = pose.shape[0]
        rot_z_90 = torch.eye(4, device=self.device)

        rot_z_90[1, 1] = -1
        rot_z_90[2, 3] = -0.001 * 79
        rot_z_90 = rot_z_90.repeat(batch_size, 1, 1)
        pose = torch.matmul(pose, rot_z_90)
        palm_vertices = self.palm.repeat(batch_size, 1, 1)
        palm_vertices = torch.matmul(pose,
                                     palm_vertices.transpose(2, 1)).transpose(
                                         1, 2)[:, :, :3]
        # The second dimension represents the number of joints for the fingers which are stored as: finger1 joint1, finger1 joint2, ..., finger 3 joint 2
        joints = torch.zeros((batch_size, 6, 4, 4), device=self.device)

        all_knuckle_vertices = torch.zeros(
            (batch_size, 2, self.knuckle.shape[0], 3), device=self.device)

        knuckle_vertices = self.knuckle.repeat(batch_size, 1, 1)

        all_finger_vertices = torch.zeros(
            (batch_size, 3, self.finger.shape[0], 3), device=self.device)
        all_finger_tip_vertices = torch.zeros(
            (batch_size, 3, self.finger_tip.shape[0], 3), device=self.device)

        finger_vertices = self.finger.repeat(batch_size, 1, 1)
        finger_tip_vertices = self.finger_tip.repeat(batch_size, 1, 1)
        for i in range(3):
            Tw1 = self.forward_kinematics(
                self.r[i] * self.Aw, torch.tensor(0, dtype=torch.float32, device=self.device), self.Dw,
                self.r[i] * theta[:, 0] - (math.pi / 2) * self.j[i],
                batch_size)
            T12 = self.forward_kinematics(self.A1, torch.tensor(math.pi / 2, dtype=torch.float32, device=self.device),
                                          0, self.phi2 + theta[:, i + 1],
                                          batch_size)
            T23 = self.forward_kinematics(self.A2, torch.tensor(math.pi, dtype=torch.float32, device=self.device), 0,
                                          self.phi3 - theta[:, i + 4], batch_size)

            if i is 0 or i is 1:
                Tw_knuckle = self.forward_kinematics(
                    self.r[(i+1) % 2] * self.Aw, torch.tensor(0, dtype=torch.float32, device=self.device), self.Dw_knuckle,
                    -1*(self.r[i] * theta[:, 0] - (math.pi / 2) * self.j[i]),
                    batch_size)
                pose_to_Tw_knuckle = torch.matmul(pose, Tw_knuckle)
                all_knuckle_vertices[:, i] = torch.matmul(pose_to_Tw_knuckle,
                                                          knuckle_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
            pose_to_T12 = torch.matmul(torch.matmul(pose, Tw1), T12)
            all_finger_vertices[:, i] = torch.matmul(
                pose_to_T12,
                finger_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
            pose_to_T23 = torch.matmul(pose_to_T12, T23)
            all_finger_tip_vertices[:, i] = torch.matmul(
                pose_to_T23,
                finger_tip_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
            joints[:, 2 * i] = pose_to_T12
            joints[:, 2 * i + 1] = pose_to_T23
        return palm_vertices, all_knuckle_vertices, all_finger_vertices, all_finger_tip_vertices, joints

    def forward_kinematics(self, A, alpha, D, theta, batch_size=1):
        c_theta = torch.cos(theta)
        s_theta = torch.sin(theta)
        c_alpha = torch.cos(alpha)
        s_alpha = torch.sin(alpha)
        l_1_to_l = torch.zeros((batch_size, 4, 4), device=self.device)
        l_1_to_l[:, 0, 0] = c_theta
        l_1_to_l[:, 0, 1] = -s_theta
        l_1_to_l[:, 0, 3] = A
        l_1_to_l[:, 1, 0] = s_theta * c_alpha
        l_1_to_l[:, 1, 1] = c_theta * c_alpha
        l_1_to_l[:, 1, 2] = -s_alpha
        l_1_to_l[:, 1, 3] = -s_alpha * D
        l_1_to_l[:, 2, 0] = s_theta * s_alpha
        l_1_to_l[:, 2, 1] = c_theta * s_alpha
        l_1_to_l[:, 2, 2] = c_alpha
        l_1_to_l[:, 2, 3] = c_alpha * D
        l_1_to_l[:, 3, 3] = 1
        return l_1_to_l
