import sys

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from net import Model
from preprocessor import InputPreprocessor, scale_to_fit

def main():
    image_file = sys.argv[1]
    debug_flag = 'debug' in sys.argv[2:3] # save intermediate results if requested
    debug_prefix = 'preprocessor' if debug_flag else None

    device = torch.device('cpu')
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # values specific to MNIST
        ])

    model = Model().to(device)
    model.load_state_dict(torch.load('parameters.pt'))
    model.eval()

    # open image and do numpy processing
    img = Image.open(image_file)
    input_image = InputPreprocessor(np.asarray(img).astype('float')/255, debug_prefix)
    input_image.process()

    # convert back to pil images and do the scaling
    pil_digits = []
    for n, digit in enumerate(input_image.digits):
        pil_digit_insert = Image.fromarray((digit*255).astype('uint8'))
        pil_digit = scale_to_fit(pil_digit_insert)
        pil_digits.append(pil_digit)
        if debug_flag:
            pil_digit.save('final%d.png'%n)

    # do the evaluation and print the result
    out = ''
    for digit in pil_digits:
        sample = transform(digit)
        sample.unsqueeze_(0)
        sample = sample.to(device)
        with torch.no_grad():
            output = model(sample)
            out += str(int(output.argmax()))
    print('Predicted result: %s'%out)

if __name__ == '__main__':
    main()
