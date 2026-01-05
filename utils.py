import matplotlib.pyplot as plt
import numpy as np

    
def view_samples(samples):
    fig, axes = plt.subplots(figsize=(14,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        img = img.detach().cpu().numpy()
        if len(img.shape) == 3:
            img = np.transpose(img, (1, 2, 0))
        else:
            dimension = int(img.shape[0] ** .5)
            img = img.reshape((dimension, dimension))
        img = ((img +1)*255 / (2)).astype(np.uint8) # rescale to pixel range (0-255)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img, cmap='Greys_r')
    plt.show()
    