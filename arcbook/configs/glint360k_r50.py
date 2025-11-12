from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

# config = edict()
# config.margin_list = (0.5, 0.4, 0.3)
# # config.margin_list = (1.0, 0.1, 0.0)
# config.network = "r18"
#
# config.resume = False
# config.output = None
# config.embedding_size = 512
# config.sample_rate = 1.0
# config.fp16 = True
# config.momentum = 0.8
# config.weight_decay = 1e-4
# config.batch_size = 500
# config.lr = 1e-5
# # config.lr = 0.0001
# config.verbose = 2000
# config.dali = False
#
# config.rec = "/media/yim/5ca43dd5-cbeb-4cae-aac0-e36cdd0808f7/book_side/mosaic_112_0703/train/"
# # config.num_classes = 180450
# # config.num_image = 5233468
# config.num_classes = 174623
# config.num_image = 5003384
# config.num_epoch = 15
# config.warmup_epoch = 0
# config.val_targets = ['val']

config = edict()
margin_list = (0.8, 0.5, 0.2)
# config.margin_list = (1.0, 0.1, 0.0)
config.network = "r18"

config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 150
config.lr = 1e-4
# config.lr = 0.0001
config.verbose = 6100
config.dali = False

config.rec = "/media/yim/5ca43dd5-cbeb-4cae-aac0-e36cdd0808f7/book_side/cleansing_book/train/"
# config.num_classes = 180450
# config.num_image = 5233468
config.num_classes = 425928
config.num_image = 1830024
config.num_epoch = 100
config.warmup_epoch = 0
config.val_targets = ['val']
