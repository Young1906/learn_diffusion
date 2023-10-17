dev: 
	python -m modules.data

exp1:
	python -m modules.exp1\
		--checkpoint logs/ecoli/version_2/checkpoints/epoch=99-step=1700.ckpt\
		--input_dim 7\
		--n_class 2\
		--z_dim 4\
		--dset ecoli

train_cvae:
	python -m modules.train_tbcvae\
		--logname ecoli\
		--batch_size 16\
		--num_workers 4\
		--pth datasets/ECOLI/ecoli.data\
		--input_dim 7\
		--n_class 2\
		--z_dim 4\
		--max_epochs 100

exp0:
	python -m modules.exp0 \
		--logname mnist\
		--batch_size 128\
		--num_workers 4\
		--z_dim 256\
		--n_class 10\
		--max_epochs 10

