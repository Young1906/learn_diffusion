dev: 
	python -m modules.data


ecoli:
	python -m modules.tabular_exp \
		--logname "ecoli"\
		--batch_size 16\
		--num_workers 4\
		--pth datasets/ECOLI/ecoli.data\
		--input_dim 7\
		--n_class 8\
		--z_dim 4\
		--max_epochs 100

cvae:
	python -m modules \
		--batch_size 128\
		--num_workers 4\
		--z_dim 256\
		--n_class 10\
		--max_epochs 10

