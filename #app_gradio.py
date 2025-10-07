# app_gradio.py

"""
Kid-friendly kiosk UI for your evaluator.
- Big Start button, movement dropdown, webcam or upload
- Returns a simple JSON result (score + summary)
- Launches locally in a browser window
"""
import os, tempfile, time
from typing import Optional
import cv2
import gradio as gr

from evaluator import evaluate, available_movements  # make sure evaluator.py is in the same folder

MODEL_HINT = "Tip: on slower PCs, use a lighter pose model (e.g., RTMPose-s)."


def record_webcam(seconds: int = 10, fps: int = 30, width: int = 960, height: int = 540) -> Optional[str]:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fn = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(fn, fourcc, fps, (width, height))
    start = time.time()
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release(); out.release()
    return fn


def run_eval(movement: str, source: str, seconds: int, video_file, model_alias: str, side: str) -> dict:
    try:
        if source == 'webcam':
            path = record_webcam(seconds=seconds)
            if path is None:
                return {"error": "Webcam not available."}
        else:
            if video_file is None:
                return {"error": "Please upload a video or choose webcam."}
            path = video_file
        result = evaluate(path, movement_key=movement, side=side, use_dtw=True, kpt_thr=0.3, model_alias=model_alias)
        return {
            "movement": result.get("movement"),
            "overall_score": round(result.get("overall_score", 0.0), 3),
            "mean_rmse": round(result.get("mean_rmse", 0.0), 4),
            "frames": result.get("frames"),
            "orientation_used": result.get("orientation_used"),
            "hint": MODEL_HINT
        }
    except Exception as e:
        return {"error": str(e)}

movements = available_movements()

with gr.Blocks(title="Taekwondo Trainer", theme=gr.themes.Soft(), css="""
/* Kiosk-friendly sizing */
#kiosk .gr-button {font-size: 26px; height: 64px;}
#kiosk .gr-dropdown, #kiosk .gr-radio, #kiosk .gr-slider {font-size: 20px;}
#kiosk .gr-video video {max-height: 360px;}
#kiosk .scorebox {font-size: 22px;}
.gr-markdown h2 {font-size: 2rem;}
""") as demo:
    gr.Markdown("## Taekwondo Trainer\nPick a move, press Start, then follow the 3-2-1 countdown.")
    with gr.Row(elem_id="kiosk"):
        movement = gr.Dropdown(choices=movements, value=movements[0], label="Movement", interactive=True)
        side = gr.Dropdown(choices=["auto","right","left"], value="auto", label="Side", interactive=True)
        source = gr.Radio(["webcam", "upload"], value="webcam", label="Input", interactive=True)
        seconds = gr.Slider(5, 20, value=10, step=1, label="Record length (s)", interactive=True)
        model_alias = gr.Dropdown(choices=["human"], value="human", label="Pose model", interactive=True)
    video = gr.Video(label="Video (used when 'upload' is selected)")
    go = gr.Button("START", elem_id="startbtn")
    out = gr.JSON(label="Result", elem_classes=["scorebox"])

    go.click(run_eval, [movement, source, seconds, video, model_alias, side], out)

if __name__ == "__main__":
    # Opens on http://127.0.0.1:7860 by default
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
```

# Save & run

* Save the first block as evaluator.py and the second as app_gradio.py in the same folder.
* Double-click app_gradio.py if you have Python file associations, or run it from a shortcut/packager as discussed.
