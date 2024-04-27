import gradio as gr
import blur_vid
import warnings
warnings.filterwarnings("ignore")

iface = gr.Interface(
    fn=blur_vid.process_and_concat_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.CheckboxGroup(choices=blur_vid.coco_labels[1:], label="Select Classes"),
        gr.Slider(label = "blur intensity",minimum = 3,maximum = 90, step = 3),
        gr.Number(label="Start Time (seconds)"),
        gr.Number(label="End Time (seconds)"),
    ],
    outputs= "video",
    title = 'BlurVista 👓'
)
iface.launch(debug =  False,inline = False)
