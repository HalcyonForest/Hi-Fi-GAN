#!g1.1
import os 
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from model.generator import Generator
# from generator_test import Generator
# from generator import Generator
from tqdm import tqdm
from model.discriminator import MultiScaleDiscriminator, MultiPeriodDiscriminator
# from discriminator_test import MultiScaleDiscriminator, MultiPeriodDiscriminator
from librosa.util import normalize
# MAX_WAV_VALUE = 32768.0

from melspec import MelSpectrogram, MelSpectrogramConfig
from config import Config
import itertools
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from loss import discriminator_loss, generator_loss, mel_loss, loss_fm
from torch.utils.data import DataLoader
from dataset import collate_fn_tensor, BufferDataset, get_data_to_buffer
from scipy.io.wavfile import read, write
import wandb


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mel_config = MelSpectrogramConfig()
mel_spectrogram = MelSpectrogram(mel_config).to(device)
cfg = Config()

buffer = get_data_to_buffer(cfg)
dataset = BufferDataset(buffer)



train_loader = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    collate_fn=collate_fn_tensor,
    drop_last=True,
    num_workers=0
)

test_melspecs = []

for audio in os.listdir(cfg.test_audio_path):
    sr, audio_arr = read(os.path.join(cfg.test_audio_path, audio))

    test_melspecs.append(mel_spectrogram(torch.from_numpy(audio_arr).unsqueeze(0).to(device)))

    


wandb.init()

generator = Generator(cfg)
MPD = MultiPeriodDiscriminator()
MSD = MultiScaleDiscriminator()

generator = generator.to(device)
MPD = MPD.to(device)
MSD = MSD.to(device)

optimizer_g = AdamW(generator.parameters(), cfg.lr, betas=(cfg.beta_1, cfg.beta_2))
optimizer_d = AdamW(itertools.chain(MSD.parameters(), MPD.parameters()), cfg.lr, betas=(cfg.beta_1, cfg.beta_2))

lr_scheduler_g = ExponentialLR(optimizer_g, gamma=cfg.lr_decay)
lr_scheduler_d = ExponentialLR(optimizer_d, gamma=cfg.lr_decay)

generator.train()
MPD.train()
MSD.train()



# log_interval = 1

ckpt_interval = 500

audio_log_interval = 250


torch.autograd.set_detect_anomaly(True)

print(device)
generator.to(device)
MPD.to(device)
MSD.to(device)
st = 0
for epoch in range(cfg.epochs):
    wandb.log({"Epoch: ": epoch})
    gan_losses = []
    disc_losses = []
    mel_losses = []
    for i, batch in enumerate(train_loader):
        st = st + 1
        # getting wav, target melspec data from batch 
        mel, audio = batch['mel'].to(device).squeeze(1), batch['audio'].to(device)
        
        audio = audio.unsqueeze(1)
        
        gen_audio = generator(mel)

        gen_mel = mel_spectrogram(gen_audio.squeeze(1))
        
        wandb.log({"Train audio: ": wandb.Audio((gen_audio).squeeze().clone().detach().cpu().numpy().astype('int16'), sample_rate=cfg.sample_rate)})


        optimizer_d.zero_grad()
        # Dealing with MultiPeriodDiscriminator

        true_mpd, gen_mpd, _, _ = MPD(audio, gen_audio.detach())
        mpd_loss = discriminator_loss(true_mpd, gen_mpd)
        
        # Dealing with MultiScaleDiscriminator
        true_msd, gen_msd, _, _ = MSD(audio, gen_audio.detach())

        msd_loss = discriminator_loss(true_msd, gen_msd)


        disc_loss = msd_loss + mpd_loss
        wandb.log({"step MSD Discriminator loss": msd_loss.item()})
        wandb.log({"step MPD Discriminator loss": mpd_loss.item()})

        disc_losses.append(disc_loss.item())

        disc_loss.backward()
        optimizer_d.step()

        # Dealing with generator

        optimizer_g.zero_grad()
        
        true_mpd, gen_mpd, mpd_fm_gt, mpd_fm_gen = MPD(audio, gen_audio)
        true_msd, gen_msd, msd_fm_gt, msd_fm_gen = MSD(audio, gen_audio)

#         loss_fm_mpd = loss_fm(mpd_fm_gt, mpd_fm_gen)
#         loss_fm_msd = loss_fm(msd_fm_gt, msd_fm_gen)
        
        loss_gen_mpd = generator_loss(gen_mpd)
        loss_gen_msd = generator_loss(gen_msd)
        

        loss_mel = F.l1_loss(mel, gen_mel)

#         loss_gen = loss_gen_mpd + loss_gen_msd + loss_fm_mpd + loss_fm_msd + loss_mel
        loss_gen = loss_gen_mpd + loss_gen_msd + loss_mel

        gan_losses.append(loss_gen.item())
        mel_losses.append(loss_mel.item())
        
        wandb.log({"step": st})
        wandb.log({"step Discriminator loss": disc_loss.item()})
        wandb.log({"step Generator loss": loss_gen.item()})
#         wandb.log({"step Generator FM loss": loss_fm_mpd.item() + loss_fm_msd.item()})
        wandb.log({"step Generator MSD MFD loss": loss_gen_mpd.item() + loss_gen_msd.item()})
        wandb.log({"step Mel loss: ":loss_mel.item() })

        loss_gen.backward()
        optimizer_g.step()
        
        if st % audio_log_interval == 0:
            k = 0
            j = 0
            generator.eval()
            with torch.no_grad():

                test_audios = []
                for test_melspec in test_melspecs:
                    print("test melspec shape: ", test_melspec.shape)
                    test_audio = generator(test_melspec).cpu()
                    write(f"test_generated/generated_{st}_{k}.wav", cfg.sample_rate, test_audio.numpy())
                    k += 1
                    test_audios.append(wandb.Audio(test_audio[0][0], sample_rate=cfg.sample_rate))
                wandb.log({"Test audios: ": test_audios})
            generator.train()
                
        if st % ckpt_interval == 0:

            torch.save(generator, os.path.join(cfg.checkpoint_path, f"Generator_{st}.pt"))
            torch.save(MPD, os.path.join(cfg.checkpoint_path, f"MPD_{st}.pt"))
            torch.save(MSD, os.path.join(cfg.checkpoint_path, f"MSD_{st}.pt"))


    
    wandb.log({"Discriminator loss": sum(disc_losses) / len(disc_losses)})
    wandb.log({"Generator loss": sum(gan_losses) / len(gan_losses)})
    wandb.log({"Mel loss: ": sum(mel_losses) / len(mel_losses)})

#     wandb.log({"Gen, GT Melspecs: ": [wandb.Image(gen_mel.reshape((-1, gen_mel.shape[-1]))), wandb.Image(mel.reshape((-1, mel.shape[-1])))]})


    lr_scheduler_d.step()
    lr_scheduler_g.step()