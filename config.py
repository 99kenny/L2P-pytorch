class Config:
    def __init__(self):
        self.num_class = 10
        self.batch_size = 8
        self.img_size = 224
        self.patch_size = 16
        self.in_channels = 3
        self.embed_dim = 768
        self.depth = 12
        self.num_head = 12

        self.datasets = "CIFAR10"
        self.data_path = "C:/Windows/System32/L2P-pytorch/file"
        self.summary_path = "C:/Windows/System32/L2P-pytorch/summary"

        self.lr = 0.001
        self.lr_momentum = 0.9
        self.weight_decay = 5e-4

        self.start_epoch = 0
        self.epochs = 100
    