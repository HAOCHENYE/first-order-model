import subprocess

import gradio as gr
import imageio
import numpy as np
import torch
import yaml
from mmengine.device import get_device
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm

from animate import normalize_kp
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from sync_batchnorm import DataParallelWithCallback

device = get_device()

def load_checkpoints(config_path, checkpoint_path):

    with open(config_path) as f:
        config = yaml.load(f, yaml.FullLoader)

    generator = OcclusionAwareGenerator(
        **config["model_params"]["generator_params"], **config["model_params"]["common_params"]
    )
    generator.to(device=device)

    kp_detector = KPDetector(**config["model_params"]["kp_detector_params"], **config["model_params"]["common_params"])
    kp_detector.to(device=device)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device=device))

    generator.load_state_dict(checkpoint["generator"])
    kp_detector.load_state_dict(checkpoint["kp_detector"])

    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def make_animation(
    source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True
):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.to(device=device)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            driving_frame = driving_frame.to(device=device)
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(
                kp_source=kp_source,
                kp_driving=kp_driving,
                kp_driving_initial=kp_driving_initial,
                use_relative_movement=relative,
                use_relative_jacobian=relative,
                adapt_movement_scale=adapt_movement_scale,
            )
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def inference(video, image):
    # trim video to 8 seconds
    source_image = imageio.imread(image)
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()["fps"]
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    predictions = make_animation(
        source_image,
        driving_video,
        generator,
        kp_detector,
        relative=True,
        adapt_movement_scale=True,
    )
    imageio.mimsave("result.mp4", [img_as_ubyte(frame) for frame in predictions], fps=fps)
    imageio.mimsave("driving.mp4", [img_as_ubyte(frame) for frame in driving_video], fps=fps)
    cmd = f"ffmpeg -y -i result.mp4 -i {video} -c copy -map 0:0 -map 1:1 -shortest out.mp4"
    subprocess.run(cmd.split())
    cmd = "ffmpeg -y -i driving.mp4 -i out.mp4 -filter_complex hstack=inputs=2 final.mp4"
    subprocess.run(cmd.split())
    return "final.mp4"


title = "First Order Motion Model"
description = "Gradio demo for First Order Motion Model. Read more at the links below. Upload a video file (cropped to face), a facial image and have fun :D. Please note that your video will be trimmed to first 8 seconds."
article = "<p style='text-align: center'><a href='https://papers.nips.cc/paper/2019/file/31c0b36aef265d9221af80872ceb62f9-Paper.pdf' target='_blank'>First Order Motion Model for Image Animation</a> | <a href='https://github.com/AliaksandrSiarohin/first-order-model' target='_blank'>Github Repo</a></p>"
examples = [["data/hurt1.mp4", "data/chairman.png"]]
generator, kp_detector = load_checkpoints(
    config_path="config/vox-256.yaml",
    checkpoint_path="checkpoints/vox-adv-cpk.pth.tar",
)

iface = gr.Interface(
    inference,
    [
        gr.inputs.Video(type="mp4"),
        gr.inputs.Image(type="filepath"),
    ],
    outputs=gr.outputs.Video(label="Output Video"),
    examples=examples,
    enable_queue=True,
    title=title,
    article=article,
    description=description,
)
iface.launch(debug=True)