class Config():
    def __init__(self):
        self.batch_size = 16
        self.epochs = 5000
        self.lr = 2e-4
        self.beta_1 = 0.8
        self.beta_2 = 0.99
        self.lr_decay = 0.999


        self.upsample = [8,8,2,2]
        self.k_u = [16,16,4,4]
        self.dim = 512
        self.k_r = [3, 7, 11]
        self.D_r = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        
        # дебаг: 
        self.resblock = '1'
        self.upsample_rates= [8,8,2,2]
        self.upsample_kernel_sizes= [16,16,4,4]
        self.upsample_initial_channel= 512
        self.resblock_kernel_sizes= [3,7,11]
        self.resblock_dilation_sizes= [[1,3,5], [1,3,5], [1,3,5]]

        self.n_mels = 80
        self.n_fft = 1024
        self.n_freq = 1025
        self.segment_len = 8192
        self.hop_len = 256
        self.win_len = 1024

        self.sample_rate = 22050

        self.f_min = 0
        self.f_max = 8000
        self.fmax_loss = None
        self.n_workers=0

        self.checkpoint_path = "./model_checkpoint"
        self.logger_path = "./logger"
        self.mel_ground_truth = "./mels"
        self.alignment_path = "./alignments"
        self.wav_path = "./data/LJSpeech-1.1/wavs"
        self.data_path = "./data/train.txt"
        self.mels_path = "./mels"

        self.test_audio_path = "./test_audios"
        self.test_mels_path = "./test_melspecs"
        self.test_generated_path = "./test_generated"


