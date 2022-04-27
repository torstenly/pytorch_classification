# USAGE
# python inference.py --model output/warmup_model.pth [--image_source path_to_images]
# python inference.py --model output/finetune_model.pth

# import the necessary packages
from pyimagesearch import config
from pyimagesearch import create_dataloaders
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
from datetime import datetime as dt
import argparse
import torch
import json
import os
import hashlib
from tqdm import tqdm

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="path to trained model model")
ap.add_argument("-i", "--image_source", required=False,
    default=config.VAL,
    help="optional path to images for inference")
args = vars(ap.parse_args())

# build our data pre-processing pipeline
testTransform = transforms.Compose([
    transforms.CenterCrop(128),         # Resize to model source tile size
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])

# calculate the inverse mean and standard deviation
invMean = [-m/s for (m, s) in zip(config.MEAN, config.STD)]
invStd = [1/s for s in config.STD]

# define our de-normalization transform
deNormalize = transforms.Normalize(mean=invMean, std=invStd)


def read_classes() -> dict:
    try:
        # read class list to model meta file
        model_fn, model_extension = os.path.splitext(args['model'])
        meta_fn = model_fn + '.json'
        with open(meta_fn) as f:
            json_data = json.load(f)

        return json_data
    except FileNotFoundError:
        print(f'File not found: {meta_fn}')


def inference():
    # initialize our test dataset and data loader
    print(f'[INFO] loading the dataset from {args["image_source"]}...')
    (testDS, testLoader) = create_dataloaders.get_dataloader(args["image_source"],
        transforms=testTransform, batchSize=config.PRED_BATCH_SIZE,
        shuffle=False)

    # read class list to model meta file
    classes = read_classes()['classes']
    print(f'Classes: {classes}')

    # report MD5 of model file
    with open(args["model"], 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()

    # write header to predictions file
    dt_string = f'{dt.now().strftime("%Y%m%d%H%M")}'
    pred_fn = os.path.join("output", f"predictions_{dt_string}.txt")
    with open(pred_fn, 'w') as f:
        f.write(f'# {dt_string}\n')
        f.write(f'# images: {args["image_source"]}\n')
        f.write(f'# model: {args["model"]}, md5: {md5}\n')
        f.write(f'# classes: {classes}\n')

    # if directory structure doesn't match model classes
    # then we have unlabeled data
    labeled_data = classes == testDS.classes
    if not labeled_data:
        print(f"Infering on unlabeled images from dirs: {testDS.classes}")

    # check if we have a GPU available, if so, define the map location
    # accordingly
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()

    # otherwise, we will be using CPU to run our model
    else:
        map_location = "cpu"

    # load the model
    print("[INFO] loading the model...")
    model = torch.load(args["model"], map_location=map_location)

    # move the model to the device and set it in evaluation mode
    model.to(config.DEVICE)
    model.eval()

    # initialize a figure
    fig = plt.figure("Results", figsize=(10, 10))

    nrows = max(1, min(5, config.PRED_LIMIT // 5))
    ncols = max(1, config.PRED_LIMIT // nrows)
    if nrows * ncols < config.PRED_LIMIT:
        ncols += 1

    # switch off autograd
    with torch.no_grad():
        print(f"[INFO] performing inference on {len(testLoader) * config.PRED_BATCH_SIZE} images")
        pred_count = 0

        for j, (images, labels, fns) in enumerate(tqdm(testLoader)):
 
            # send the images to the device
            images = images.to(config.DEVICE)

            # make the predictions
            preds = model(images)

            # loop over all the batch
            for i in range(0, len(images)):
                # grab the image, de-normalize it, scale the raw pixel
                # intensities to the range [0, 255], and change the channel
                # ordering from channels first tp channels last
                image = images[i]
                image = deNormalize(image).cpu().numpy()
                image = (image * 255).astype("uint8")
                image = image.transpose((1, 2, 0))

                # grab the ground truth label
                idx = labels[i].cpu().numpy()

                # grab the predicted label
                pred = preds[i].argmax().cpu().numpy()
                predLabel = classes[pred]
                probs = torch.nn.functional.softmax(preds[i], dim=-1)

                if (probs[pred] >= config.PRED_MIN_CONF and
                    (config.PRED_LIMIT == 0 or pred_count < config.PRED_LIMIT) and
                    predLabel not in config.PRED_EXCL_CLASSES):

                    if 0 < config.PRED_LIMIT:
                        # initalize a subplot
                        plt.subplot(nrows, ncols, pred_count + 1)

                        # grab the ground truth label
                        gt_idx = labels[i].cpu().numpy()

                        # add the results and image to the plot
                        gtLabel = f'[GT: {classes[gt_idx]}]' if labeled_data and gt_idx != pred else ''
                        info = (f"{predLabel} ({probs[pred]:0.03}) {gtLabel}")
                        plt.imshow(image)
                        plt.title(info)
                        plt.axis("off")

                    # output prediction to file
                    with open(pred_fn, 'a') as f:
                        f.write(f'{fns[i]},{predLabel},{probs[pred]:.5f}\n')

                    pred_count += 1

            if config.PRED_LIMIT > 0 and pred_count >= config.PRED_LIMIT:
                break

    # show the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    inference()

