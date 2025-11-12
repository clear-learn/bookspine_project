from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r100"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 128
config.lr = 0.1
config.verbose = 2000
config.dali = False

config.rec = "/media/yimhaksoon/5ca43dd5-cbeb-4cae-aac0-e36cdd0808f7/book_side/112_publish/train"
config.num_classes = 514094
config.num_image = 907470
config.num_epoch = 200
config.warmup_epoch = 20
config.val_targets = ['val']
