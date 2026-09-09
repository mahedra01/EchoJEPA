import cv2
import numpy as np
import torch

from pathlib import Path


ImageNet_Mean = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

ImageNet_Std = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def Load_And_Preprocess_Video(
    Video_Path,
    Number_Of_Frames=16,
    Image_Size=224
):

    Video_Path = Path(
        Video_Path
    )

    if not Video_Path.exists():

        raise FileNotFoundError(
            f"Video not found: {Video_Path}"
        )


    Video_Capture = cv2.VideoCapture(
        str(Video_Path)
    )

    if not Video_Capture.isOpened():

        raise RuntimeError(
            f"Cannot open video: {Video_Path}"
        )


    Frame_Count = int(
        Video_Capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if Frame_Count <= 0:

        Video_Capture.release()

        raise RuntimeError(
            f"Invalid frame count: {Video_Path}"
        )


    Frame_Indices = np.linspace(
        0,
        Frame_Count - 1,
        Number_Of_Frames,
        dtype=int
    )


    Processed_Frames = []


    for Frame_Index in Frame_Indices:

        Video_Capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(Frame_Index)
        )

        Success, Frame = (
            Video_Capture.read()
        )

        if not Success:

            Video_Capture.release()

            raise RuntimeError(
                f"Cannot read frame "
                f"{Frame_Index} from "
                f"{Video_Path}"
            )


        Frame_RGB = cv2.cvtColor(
            Frame,
            cv2.COLOR_BGR2RGB
        )


        Gray_Frame = cv2.cvtColor(
            Frame_RGB,
            cv2.COLOR_RGB2GRAY
        )


        Resized_Gray_Frame = cv2.resize(
            Gray_Frame,
            (
                Image_Size,
                Image_Size
            ),
            interpolation=cv2.INTER_LINEAR
        )


        Frame_3Channel = np.repeat(
            Resized_Gray_Frame[
                :,
                :,
                np.newaxis
            ],
            3,
            axis=2
        )


        Processed_Frames.append(
            Frame_3Channel
        )


    Video_Capture.release()


    Processed_Frames = np.stack(
        Processed_Frames,
        axis=0
    )


    Normalized_Frames = (
        Processed_Frames.astype(
            np.float32
        )
        /
        255.0
    )


    Normalized_Frames = (
        Normalized_Frames
        -
        ImageNet_Mean
    ) / ImageNet_Std


    Video_Tensor = torch.from_numpy(
        Normalized_Frames
    )


    Video_Tensor = (
        Video_Tensor
        .permute(
            3,
            0,
            1,
            2
        )
        .contiguous()
    )


    Expected_Shape = (
        3,
        Number_Of_Frames,
        Image_Size,
        Image_Size
    )


    if Video_Tensor.shape != Expected_Shape:

        raise RuntimeError(
            f"Unexpected video shape: "
            f"{Video_Tensor.shape}"
        )


    if not torch.isfinite(
        Video_Tensor
    ).all():

        raise RuntimeError(
            f"NaN or Inf found in: "
            f"{Video_Path}"
        )


    return (
        Video_Tensor,
        Frame_Indices
    )
