import numpy as np
import matplotlib.pyplot as plt

def show_images(images, labels=[], compact=True):
    """
    """
    img_size = images[0].shape[:2]
    n = len(images)
    columns = np.round(np.sqrt(n / 9 * 16))
    rows = np.round(n / columns + 0.5)
    
    cmap = None
    if len(images[0].shape) == 3 and images[0].shape[-1] == 1:
        images = np.squeeze(images, -1)
        cmap = 'gray'

    fig = plt.figure()
    if compact:
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i in range(n):
        subPlt = fig.add_subplot(rows, columns, i + 1)
        if compact:
            subPlt.axis('off')
        plt.imshow(images[i].astype('uint8'), cmap)
    plt.show(block=True) 
