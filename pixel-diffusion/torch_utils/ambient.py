import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import torch.nn as nn
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
import ambient_utils
import wandb
from ambient_utils.classifier import analyze_classifier_trajectory
from collections import defaultdict
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import pandas as pd

def apply_ema(data, window=3):
    return pd.Series(data).ewm(span=window, adjust=False).mean().values

def load_annotations(dataset_path, cls_ema_window=32, cls_epsilon=0.05):  
  # check if there is a file annotations.jsonl in the dataset_kwargs.path
  annotations_file = os.path.join(dataset_path, "annotations.jsonl")
  annotations = defaultdict(lambda: (0., 0.))
  lines_read = 0
  if os.path.exists(annotations_file):
      # read sigmas from the file
      sigmas_path = os.path.join(dataset_path, "sigmas.txt")
      # make sure that the the filepath exists
      if os.path.exists(sigmas_path):
          with open(sigmas_path, "r") as f:
              sigmas = [float(line.strip()) for line in f]
          device = "cuda" if torch.cuda.is_available() else "cpu"
          sigmas = torch.tensor(sigmas, device=device)
          sigmas = sigmas.sort(dim=0)[0]
      with open(annotations_file, "r") as f:
          for line in f:
              lines_read += 1
              line_json = json.loads(line)
              filename = line_json["filename"]
              # raw probabilities are stored that need to be processed
              if "probabilities" in line_json:
                  probs = line_json["probabilities"]
                  probs = np.array(probs).mean(axis=-1)
                  ema_probs = apply_ema(probs, window=cls_ema_window)
                  first_confusion = analyze_classifier_trajectory(torch.tensor(ema_probs).to(device), sigmas, epsilon=cls_epsilon)['first_confusion']
                  annotations[filename] = (first_confusion.cpu().item(), 0.)
              elif any(key.startswith("crop_predictions") for key in line_json):
                  patch_size_to_probs = {}
                  for key, value in line_json.items():
                      if key.startswith("crop_predictions"):
                          patch_size = int(key.split("_")[-1])
                          patch_size_to_probs[patch_size] = np.mean(value)
                  
                  # get the biggest crop size for which the probability is above 0.3
                  for patch_size in sorted(patch_size_to_probs.keys(), reverse=True):
                      if patch_size_to_probs[patch_size] > 0.25:
                          break
                  else:
                      patch_size = 1
                  
                  patch_to_sigma = {
                      1: 0.01,
                      4: 0.05,
                      8: 0.15,
                      16: 0.2,
                      24: 0.35,
                      32: 0.55,
                      48: 0.7,
                      64: 1.0,
                  }
                  sigma_max = patch_to_sigma[patch_size]
                  annotations[filename] = (300.0, sigma_max)

              # if single time
              elif "annotation" in line_json or "sigma" in line_json:
                  annotations[filename] = (line_json["annotation"], 0.) if "annotation" in line_json else (line_json["sigma"], 0)
              elif "sigma_min" in line_json and "sigma_max" in line_json:
                  annotations[filename] = (line_json["sigma_min"], line_json["sigma_max"])
              else:
                  raise ValueError(f"Could not parse line {line}")

  # print the number of annotations
  print(f"Num annotations: {len(list(annotations.keys()))}, Lines read: {lines_read}")
  # print the average min annotation
  print(f"Average min annotation: {np.mean([x[0] for x in annotations.values()])}")
  # print the average min annotation excluding values that are exactly 0
  print(f"Average min annotation excluding 0: {np.mean([x[0] for x in annotations.values() if (x[0] != 0 and x[0] != 300)])}")
  # print the average max annotation
  print(f"Average max annotation: {np.mean([x[1] for x in annotations.values()])}")
  # print the average max annotation excluding values that are exactly 0
  print(f"Average max annotation excluding 0: {np.mean([x[1] for x in annotations.values() if x[1] != 0])}")

  return annotations