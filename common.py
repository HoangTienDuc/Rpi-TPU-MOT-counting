# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utilities."""
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import platform

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

# def set_batch_input(interpreter, images, resample=Image.NEAREST):
#     """Transforms list of different sized images to input tensor --- no can do, edge tpu doesnt work like that"""
#     reshaped_images = []  
#     for image in images:
#       if 0 in image.shape:
#         continue
#       iis_full = input_image_size(interpreter)
#       print("iis full je ")
#       print(iis_full)
#       iis = iis_full[0:2]
#       print("common set batch input take two {}".format(iis))

#       reshaped_image = Image.fromarray(image).resize((iis), resample)
#       reshaped_images.append(np.asarray(reshaped_image))
#     reshaped_images = np.array(reshaped_images)
#     print(reshaped_images.shape)
#     #nezz jel moze ovak il moze sam jedna slika kao input...
#     input_tensor(interpreter)[:, :] = reshaped_images

def set_input(interpreter, image, resample=Image.NEAREST):
    """Copies data to input tensor."""

    image = image.resize((input_image_size(interpreter)[0:2]), resample)
    input_tensor(interpreter)[:, :] = image

def input_image_size(interpreter):
    """Returns input image size as (width, height, channels) tuple."""
    _, height, width, channels = interpreter.get_input_details()[0]['shape']
    return width, height, channels

def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]

def output_tensor(interpreter, i):
    """Returns dequantized output tensor if quantized before."""
    output_details = interpreter.get_output_details()[i]
    output_data = np.squeeze(interpreter.tensor(output_details['index'])())
    if 'quantization' not in output_details:
        return output_data
    scale, zero_point = output_details['quantization']
    if scale == 0:
        return output_data - zero_point
    return scale * (output_data - zero_point)
