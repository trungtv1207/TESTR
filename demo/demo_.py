# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.data.detection_utils import read_image

from predictor import VisualizationDemo
from adet.config import get_cfg

# constants
WINDOW_NAME = "COCO detections"

import gradio as gr

def setup_cfg(config_file, opts, confidence_threshold, inference_th_test):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file("./configs/TESTR/ICDAR15/TESTR_R_50_Polygon.yaml")
    cfg.merge_from_list(opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.MODEL.WEIGHTS = "../weights/pretrain_testr_R_50_polygon.pth"
    cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST = inference_th_test
    cfg.freeze()
    return cfg

def predict(input, config_file, input_type, confidence_threshold, inference_th_test, opts=[], progress=gr.Progress()):
    progress(0, desc="Starting...")
    output = "./datasets/demo_"
    gr.Info("Starting process")
    
    if config_file != "ICDAR15":
        raise gr.Error("Please select ICDAR15 in Config File!")
    if input_type != "input":
        raise gr.Error("Please select input in Input Type!")
    if input is None:
        raise gr.Error("Please select image in Input Image!")
    else:
        input = [input]

    mp.set_start_method("spawn", force=True)

    cfg = setup_cfg(config_file, opts, confidence_threshold, inference_th_test)

    demo = VisualizationDemo(cfg)
    if input:
        if os.path.isdir(input[0]):
            input = [os.path.join(input[0], fname) for fname in os.listdir(input[0])]
        elif len(input) == 1:
            input = glob.glob(os.path.expanduser(input[0]))
            assert input, "The input path(s) was not found"
        for path in tqdm.tqdm(input, disable=not output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output, visualized_results = demo.run_on_image(img)
            print(visualized_results)
            if output:
                if os.path.isdir(output):
                    assert os.path.isdir(output), output
                    out_filename = os.path.join(output, os.path.basename(path))
                else:
                    assert len(input) == 1, "Please specify a directory with output"
                    out_filename = output
                visualized_output.save(out_filename)
            else:
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    return out_filename, visualized_results

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(
                label='Input Image', 
                type = "filepath",
                # image_mode="RGB"
            ),
            gr.Dropdown(
                label='Config File', 
                # info='In years, must be greater than 0', 
                choices=["Pretrain", "TotalText", "CTW1500", "ICDAR15", "Pretrain (Lite)", "TotalText (Lite)"],
                value="ICDAR15"
            ),
            gr.Dropdown(
                label='Input Type', 
                # info='In years, must be greater than 0', 
                choices=["webcam", "video_input", "input"],
                value="input"
            ),
            gr.Slider(
                label='Confidence Threshold', 
                # info='In years, must be greater than 0', 
                minimum=0, 
                maximum=1, 
                value=0.3
            ),
            gr.Slider(
                label='Inference Th Test', 
                # info='In years, must be greater than 0', 
                minimum=0, 
                maximum=1, 
                value=0.3
            ),
            gr.Text(
                label="Opts"
            )
            ],
    outputs=[gr.Image(
                label='Output Image', 
            ),
            gr.JSON(
                label="Output Results")
            ],
    title="TESTR",
    description="Text Spotting Transformers",
    article="Text Spotting Transformers",
    theme=gr.themes.Monochrome(),
    # batch=True, 
    # max_batch_size=16
)

if __name__ == "__main__":
    # demo.queue()
    demo.launch(show_api=False, share=True, auth=("hoanganhh", "hoanganhh"))
    
