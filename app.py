from transformers import pipeline, SamModel, SamProcessor
import torch
import numpy as np
import gradio as gr
import spaces

checkpoint = "google/owlv2-base-patch16-ensemble"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device="cuda")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda")
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


@spaces.GPU
def query(image, texts, threshold, sam_threshold):
  texts = texts.split(",")

  predictions = detector(
    image,
    candidate_labels=texts,
    threshold=threshold
  )

  result_labels = []
  for pred in predictions:

    box = pred["box"]
    score = pred["score"]
    label = pred["label"]
    box = [round(pred["box"]["xmin"], 2), round(pred["box"]["ymin"], 2),
        round(pred["box"]["xmax"], 2), round(pred["box"]["ymax"], 2)]
    inputs = sam_processor(
            image,
            input_boxes=[[box]],
            return_tensors="pt"
        ).to("cuda")
    with torch.no_grad():
      outputs = sam_model(**inputs)

    mask = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )
    iou_scores = outputs["iou_scores"]
    
    masks, iou_scores, boxes = sam_processor.image_processor.filter_masks(
        mask[0],
        iou_scores[0].cpu(),
        inputs["original_sizes"][0].cpu(),
        box,
        pred_iou_thresh=sam_threshold,
    )
    
    result_labels.append((mask[0][0][0].numpy(), label))
  return image, result_labels


description = "This Space combines OWLv2, the state-of-the-art zero-shot object detection model with SAM, the state-of-the-art mask generation model. SAM normally doesn't accept text input. Combining SAM with OWLv2 makes SAM text promptable. Try the example or input an image and comma separated candidate labels to segment."
demo = gr.Interface(
    query,
    inputs=[gr.Image(type="pil", label="Image Input"), gr.Textbox(label = "Candidate Labels"), gr.Slider(0, 1, value=0.05, label="Confidence Threshold for OWL"), gr.Slider(0, 1, value=0.88, label="IoU threshold for SAM")],
    outputs="annotatedimage",
    title="OWL 🤝 SAM",
    description=description,
    examples=[
        ["./cats.png", "cat", 0.1, 0.88],
    ],
    cache_examples=True
)
demo.launch(debug=True)