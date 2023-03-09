#!/usr/bin/env python

import os
import joblib
import argparse
from PIL import Image
from .util import draw_bb_on_img
from .constants import MODEL_PATH
from face_recognition import preprocessing
import pickle as pkl


def parse_args():
    parser = argparse.ArgumentParser(
        'Script for detecting and classifying faces on user-provided image. This script will process image, draw '
        'bounding boxes and labels on image and display it. It will also optionally save that image.')
    parser.add_argument('--test-metadata', required=True, help='Test metadata')
    parser.add_argument('--save-dir', help='If save dir is provided image will be saved to specified directory.')
    return parser.parse_args()


def recognise_faces(img):
    faces = joblib.load(MODEL_PATH)(img)
    if faces:    
        return faces[0].top_prediction.label.upper()
    else:
        return "None_face"


def main():
    args = parse_args()
    with open(args.test_metadata,'rb') as f:
        test_metadata = pkl.load(f)

    preprocess = preprocessing.ExifOrientationNormalize()

    total=0
    correct=0
    for name,data_list in test_metadata.items():
        for data in data_list:
            img = Image.open(data["path"])
            img = preprocess(img).convert('RGB')
            pred = recognise_faces(img)
            print(f"{pred} - {name}")
            total +=1
            if pred.lower()==name.lower():
                correct += 1

    print(f"Accuracy: {correct/(total)}")


if __name__ == '__main__':
    main()
