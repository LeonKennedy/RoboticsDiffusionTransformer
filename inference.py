import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
import cv2

from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.rdt_runner import RDTRunner
from lerobot.devices import build_right_arm, CameraGroup, build_robot
from configs.state_vec import STATE_VEC_IDX_MAPPING

from data.qkids.qkids_stat import Normalizer

observation_window = deque(maxlen=2)

lang_embeddings = None

# debug
preload_images = None

AGILEX_STATE_INDICES = [
    [
        STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
    ] + [
        STATE_VEC_IDX_MAPPING[f"right_gripper_open"]
    ]]


class RoboticDiffusionTransformerModel(object):
    """A wrapper for the RDT model, which handles
            1. Model initialization
            2. Encodings of instructions
            3. Model inference
    """

    def __init__(
            self, args,
            device='cuda',
            dtype=torch.bfloat16,
            image_size=None,
            control_frequency=25,
            pretrained=None,
            pretrained_vision_encoder_name_or_path=None,
    ):
        self.args = args
        self.dtype = dtype
        self.image_size = image_size
        self.device = device
        self.control_frequency = control_frequency
        # We do not use the text encoder due to limited GPU memory
        # self.text_tokenizer, self.text_model = self.get_text_encoder(pretrained_text_encoder_name_or_path)
        self.image_processor, self.vision_model = self.get_vision_encoder(pretrained_vision_encoder_name_or_path)
        self.policy = self.get_policy(pretrained)

        self.reset()

    def get_policy(self, pretrained):
        """Initialize the model."""
        # Initialize model with arguments
        if (
                pretrained is None
                or os.path.isfile(pretrained)
        ):
            img_cond_len = (self.args["common"]["img_history_size"]
                            * self.args["common"]["num_cameras"]
                            * self.vision_model.num_patches)

            _model = RDTRunner(
                action_dim=self.args["common"]["state_dim"],
                pred_horizon=self.args["common"]["action_chunk_size"],
                config=self.args["model"],
                lang_token_dim=self.args["model"]["lang_token_dim"],
                img_token_dim=self.args["model"]["img_token_dim"],
                state_token_dim=self.args["model"]["state_token_dim"],
                max_lang_cond_len=self.args["dataset"]["tokenizer_max_length"],
                img_cond_len=img_cond_len,
                img_pos_embed_config=[
                    # No initial pos embed in the last grid size
                    # since we've already done in ViT
                    ("image", (self.args["common"]["img_history_size"],
                               self.args["common"]["num_cameras"],
                               -self.vision_model.num_patches)),
                ],
                lang_pos_embed_config=[
                    # Similarly, no initial pos embed for language
                    ("lang", -self.args["dataset"]["tokenizer_max_length"]),
                ],
                dtype=self.dtype,
            )
        else:
            _model = RDTRunner.from_pretrained(pretrained)

        return _model

    def get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
        vision_encoder = SiglipVisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None)
        image_processor = vision_encoder.image_processor
        return image_processor, vision_encoder

    def reset(self):
        """Set model to evaluation mode.
        """
        device = self.device
        weight_dtype = self.dtype
        self.policy.eval()
        # self.text_model.eval()
        self.vision_model.eval()

        self.policy = self.policy.to(device, dtype=weight_dtype)
        # self.text_model = self.text_model.to(device, dtype=weight_dtype)
        self.vision_model = self.vision_model.to(device, dtype=weight_dtype)

    def load_pretrained_weights(self, pretrained=None):
        if pretrained is None:
            return
        print(f'Loading weights from {pretrained}')
        filename = os.path.basename(pretrained)
        if filename.endswith('.pt'):
            checkpoint = torch.load(pretrained)
            self.policy.load_state_dict(checkpoint["module"])
        elif filename.endswith('.safetensors'):
            from safetensors.torch import load_model
            load_model(self.policy, pretrained)
        else:
            raise NotImplementedError(f"Unknown checkpoint format: {pretrained}")

    def _format_joint_to_state(self, joints):
        """
        Format the joint proprioception into the unified action vector.

        Args:
            joints (torch.Tensor): The joint proprioception to be formatted.
                qpos ([B, N, 14]).

        Returns:
            state (torch.Tensor): The formatted vector for RDT ([B, N, 128]).
        """
        # Rescale the gripper to the range of [0, 1]
        # joints = joints / torch.tensor(
        #     [[[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]]],
        #     device=joints.device, dtype=joints.dtype
        # )

        B, N, _ = joints.shape
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]),
            device=joints.device, dtype=joints.dtype
        )
        # Fill into the unified state vector
        state[:, :, AGILEX_STATE_INDICES] = joints
        # Assemble the mask indicating each dimension's availability
        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=joints.device, dtype=joints.dtype
        )
        state_elem_mask[:, AGILEX_STATE_INDICES] = 1
        return state, state_elem_mask

    def _unformat_action_to_joint(self, action):
        """
        Unformat the unified action vector into the joint action to be executed.

        Args:
            action (torch.Tensor): The unified action vector to be unformatted.
                ([B, N, 128])

        Returns:
            joints (torch.Tensor): The unformatted robot joint action.
                qpos ([B, N, 14]).
        """
        action_indices = AGILEX_STATE_INDICES
        joints = action[:, :, action_indices]

        # Rescale the gripper back to the action range
        # Note that the action range and proprioception range are different
        # for Mobile ALOHA robot

        return joints

    @torch.no_grad()
    def step(self, proprio, images, text_embeds):
        """
        Predict the next action chunk given the
        proprioceptive states, images, and instruction embeddings.

        Args:
            proprio: proprioceptive states
            images: RGB images, the order should be
                [ext_{t-1}, right_wrist_{t-1}, left_wrist_{t-1},
                ext_{t}, right_wrist_{t}, left_wrist_{t}]
            text_embeds: instruction embeddings

        Returns:
            action: predicted action
        """
        device = self.device
        dtype = self.dtype

        # The background image used for padding
        background_color = np.array([
            int(x * 255) for x in self.image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)
        background_image = np.ones((
            self.image_processor.size["height"],
            self.image_processor.size["width"], 3), dtype=np.uint8
        ) * background_color

        # Preprocess the images by order and encode them
        image_tensor_list = []
        for image in images:
            if image is None:
                # Replace it with the background image
                image = Image.fromarray(background_image)

            if self.image_size is not None:
                image = transforms.Resize(self.data_args.image_size)(image)

            if self.args["dataset"].get("auto_adjust_image_brightness", False):
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)

            if self.args["dataset"].get("image_aspect_ratio", "pad") == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor_list.append(image)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)

        image_embeds = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)

        # Prepare the proprioception states and the control frequency
        joints = proprio.to(device).unsqueeze(0)  # (1, 1, 14)
        states, state_elem_mask = self._format_joint_to_state(joints)  # (1, 1, 128), (1, 128)
        states, state_elem_mask = states.to(device, dtype=dtype), state_elem_mask.to(device, dtype=dtype)
        states = states[:, -1:, :]  # (1, 1, 128)
        ctrl_freqs = torch.tensor([self.control_frequency]).to(device)

        text_embeds = text_embeds.to(device, dtype=dtype)

        # Predict the next action chunk given the inputs
        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(
                text_embeds.shape[:2], dtype=torch.bool,
                device=text_embeds.device),
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),
            ctrl_freqs=ctrl_freqs
        )
        trajectory = self._unformat_action_to_joint(trajectory).to(torch.float32)

        return trajectory


def create_model(args, **kwargs):
    model = RoboticDiffusionTransformerModel(args, **kwargs)
    pretrained = kwargs.get("pretrained", None)
    if (
            pretrained is not None
            and os.path.isfile(pretrained)
    ):
        model.load_pretrained_weights(pretrained)
    return model


def make_policy(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config

    # pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=args.config,
        dtype=torch.bfloat16,
        pretrained="outs/checkpoint-57000",
        # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=25,
    )

    return model


def update_observation_window(args, config, robot, camera):

    global observation_window

    img = camera.read(['TOP', 'RIGHT'])
    img_top = img["TOP"]
    img_right = img["RIGHT"]

    puppet_state = robot.read_puppet_state()
    qpos = norm.encode(puppet_state)
    # img_front, img_left, img_right, puppet_arm_left, puppet_arm_right = get_ros_observation(args, ros_operator)

    # qpos = np.concatenate(
    #     (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
    # qpos = torch.from_numpy(qpos).float().cuda()
    one = {
        'qpos': torch.from_numpy(qpos).float().cuda(),
        'images':
            {
                config["camera_names"][0]: img_top,
                config["camera_names"][1]: img_right,
            },
    }
    observation_window.append(one)

    if len(observation_window) == 1:
        observation_window.append(one)


def inference_fn(args, config, policy, t):
    global observation_window
    global lang_embeddings

    background_color = np.array([
        int(x * 255) for x in policy.image_processor.image_mean
    ], dtype=np.uint8).reshape(1, 1, 3)
    background_image = np.ones((
        384, 384, 3), dtype=np.uint8
    ) * background_color

    # print(f"Start inference_thread_fn: t={t}")
    while True:
        time1 = time.time()

        # fetch images in sequence [front, right, left]
        image_arrs = [
            observation_window[-2]['images'][config['camera_names'][0]],
            observation_window[-2]['images'][config['camera_names'][1]],
            background_image,

            observation_window[-1]['images'][config['camera_names'][0]],
            observation_window[-1]['images'][config['camera_names'][1]],
            background_image
        ]

        # fetch debug images in sequence [front, right, left]
        # image_arrs = [
        #     preload_images[config['camera_names'][0]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][2]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][1]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][0]][t],
        #     preload_images[config['camera_names'][2]][t],
        #     preload_images[config['camera_names'][1]][t]
        # ]
        # # encode the images
        # for i in range(len(image_arrs)):
        #     image_arrs[i] = cv2.imdecode(np.frombuffer(image_arrs[i], np.uint8), cv2.IMREAD_COLOR)
        # proprio = torch.from_numpy(preload_images['qpos'][t]).float().cuda()

        images = [Image.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]

        # for i, pos in enumerate(['f', 'r', 'l'] * 2):
        #     images[i].save(f'{t}-{i}-{pos}.png')

        # get last qpos in shape [14, ]
        proprio = observation_window[-1]['qpos']
        # unsqueeze to [1, 14]
        proprio = proprio.unsqueeze(0)

        # actions shaped as [1, 64, 14] in format [left, right]
        actions = policy.step(
            proprio=proprio,
            images=images,
            text_embeds=lang_embeddings
        ).squeeze().cpu().numpy()
        # print(f"inference_actions: {actions.squeeze()}")

        print(f"Model inference time: {time.time() - time1} s")

        # print(f"Finish inference_thread_fn: t={t}")
        return actions


def interpolate_action(args, prev_action, cur_action):
    steps = np.array(args.arm_steps_length)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def make_inference(args, config):
    global lang_embeddings

    robot = build_robot(7)
    robot.move_start_position(master=False)
    camera = CameraGroup(['TOP', 'RIGHT'], 120, 160)

    # Load rdt model
    policy = make_policy(args)

    lang_dict = torch.load("outs/put_puzzle.pt")
    print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    lang_embeddings = lang_dict["embeddings"].unsqueeze(0)

    max_publish_step = config['episode_len']
    chunk_size = 30

    # Initialize position of the puppet arm

    # ros_operator.puppet_arm_publish_continuous(left0, right0)
    # input("Press enter to continue")
    # ros_operator.puppet_arm_publish_continuous(left1, right1)
    # Initialize the previous action to be the initial robot state
    pre_action = np.array(
        [32, 16, 90, -3, -86, -6, 80]
    )
    with torch.inference_mode():
        # The current time step
        t = 0

        # action_buffer = np.zeros([chunk_size, config['state_dim']])

        while t < max_publish_step:
            start = time.time()
            # Update observation window
            update_observation_window(args, config, robot, camera)

            # When coming to the end of the action chunk
            if t % chunk_size == 0:
                # Start inference
                action_buffer = inference_fn(args, config, policy, t).copy()
                print(';;;;;;;;;;;;;--------------------------------------------------------------')

            raw_action = action_buffer[t % chunk_size]
            action = raw_action
            # Interpolate the original action sequence
            if args.use_actions_interpolation:
                # print(f"Time {t}, pre {pre_action}, act {action}")
                interp_actions = interpolate_action(args, pre_action, action)
            else:
                interp_actions = action[np.newaxis, :]

            bit_width = 1 / (time.time() - start) / 2
            # Execute the interpolated actions one by one
            for i,  act in enumerate(interp_actions):
                print(act)
                act2 = norm.decode(act)
                print(act2)
                if t % chunk_size == 0:
                    robot.set_state(act2, bit_width)
                else:
                    robot.set_state(act2, 5)
                    time.sleep(0.1)
                # print(f"doing action: {act}")
            t += 1

            print("Published Step", t)
            pre_action = action.copy()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int,
                        help='Maximum number of action publishing steps', default=10000, required=False)
    parser.add_argument('--seed', action='store', type=int,
                        help='Random seed', default=None, required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)

    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)

    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)

    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store_true',
                        help='Whether to use the robot base to move around',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int,
                        help='The rate at which to publish the actions',
                        default=30, required=False)
    parser.add_argument('--ctrl_freq', action='store', type=int,
                        help='The control frequency of the robot',
                        default=25, required=False)

    parser.add_argument('--chunk_size', action='store', type=int,
                        help='Action chunk size',
                        default=64, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float,
                        help='The maximum change allowed for each joint per timestep',
                        default=[1, 1, 1, 1, 1, 1, 50], required=False)

    parser.add_argument('--use_actions_interpolation', action='store_true',
                        help='Whether to interpolate the actions if the difference is too large',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store_true',
                        help='Whether to use depth images',
                        default=False, required=False)

    parser.add_argument('--disable_puppet_arm', action='store_true',
                        help='Whether to disable the puppet arm. This is useful for safely debugging', default=False)

    parser.add_argument('--config_path', type=str, default="configs/base.yaml",
                        help='Path to the config file')

    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    args = get_arguments()
    if args.seed is not None:
        set_seed(args.seed)
    config = {
        'episode_len': args.max_publish_step,
        'state_dim': 7,
        'chunk_size': args.chunk_size,
        'camera_names': ['cam_high', 'cam_right_wrist'],
    }
    make_inference(args, config)


if __name__ == '__main__':
    norm  = Normalizer()
    main()
