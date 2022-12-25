import torch
from torch.nn import MSELoss, L1Loss
from torch.nn.functional import mse_loss, l1_loss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def generator_loss(pred_on_fake):
    # return GD_adv_loss(pred_on_fake) #+ loss_fm(fm_d, fm_g) + mel_loss(true_melspec, fake_melspec)
#     total_loss = 0
#     for i in range(len(pred_on_fake)):
#         shape_f = pred_on_fake[i].shape
#         total_loss += mse_loss(pred_on_fake[i], torch.ones(shape_f).to(device))
#     return total_loss
    loss = 0
    gen_losses = []
    for dg in pred_on_fake:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss


def discriminator_loss(pred_on_true, pred_on_fake):
    total_loss = 0
    for i in range(len(pred_on_true)):
        shape_t = pred_on_true[i].shape
        shape_f = pred_on_fake[i].shape
        total_loss = total_loss + mse_loss(pred_on_true[i], torch.ones(shape_t).to(device)) + mse_loss(pred_on_fake[i], torch.zeros(shape_f).to(device))
    return total_loss
# Проверка отдебажить код нужно было.
#     loss = 0
#     r_losses = []
#     g_losses = []
#     for dr, dg in zip(pred_on_true, pred_on_fake):
#         r_loss = torch.mean((1-dr)**2)
#         g_loss = torch.mean(dg**2)
#         loss += (r_loss + g_loss)
#         r_losses.append(r_loss.item())
#         g_losses.append(g_loss.item())

#     return total_loss


def loss_fm(fm_d, fm_g):
#     l1 = L1Loss()
#     loss = 0
#     for i in range(len(fm_d)):
#         for j in range(len(fm_d[i])):
#             loss += l1_loss(fm_d[i][j],fm_g[i][j])
#     return 2 * loss
    loss = 0
    for dr, dg in zip(fm_d, fm_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2

def mel_loss(gt_melspec, generated_melspec):
    # loss = L1Loss()
    return 45 * l1_loss(generated_melspec, gt_melspec)
