
def init_weights(model, mean=0.0, std=0.01):
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        model.weight.data.normal_(mean, std)


def compute_pad(kernel, d):
    pad_size = int((kernel - 1) * d / 2)
    return pad_size

# Для дебага
def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)