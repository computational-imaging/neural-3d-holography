"""
A script for model training

"""
import os
import configargparse
import utils
import prop_model
import params
import image_loader as loaders

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

# Command line argument processing
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
params.add_parameters(p, 'train')
opt = p.parse_args()
params.set_configs(opt)
run_id = params.run_id_training(opt)

def main():
    if ',' in opt.data_path:
        opt.data_path = opt.data_path.split(',')
    else:
        opt.data_path = opt.data_path
    print(f'  - training a model ... Dataset path:{opt.data_path}')
    # Setup up dataloaders
    num_workers = 8
    train_loader = DataLoader(loaders.PairsLoader(os.path.join(opt.data_path, 'train'),
                                                  plane_idxs=opt.plane_idxs['all'], image_res=opt.image_res,
                                                  avg_energy_ratio=opt.avg_energy_ratio, slm_type=opt.slm_type),
                              num_workers=num_workers, batch_size=opt.batch_size)
    val_loader = DataLoader(loaders.PairsLoader(os.path.join(opt.data_path, 'val'),
                                                plane_idxs=opt.plane_idxs['all'], image_res=opt.image_res,
                                                shuffle=False, avg_energy_ratio=opt.avg_energy_ratio,
                                                slm_type=opt.slm_type),
                            num_workers=num_workers, batch_size=opt.batch_size, shuffle=False)
    test_loader = DataLoader(loaders.PairsLoader(os.path.join(opt.data_path, 'test'),
                                                 plane_idxs=opt.plane_idxs['all'], image_res=opt.image_res,
                                                 shuffle=False, avg_energy_ratio=opt.avg_energy_ratio, slm_type=opt.slm_type),
                             num_workers=num_workers, batch_size=opt.batch_size, shuffle=False)

    # Init model
    model = prop_model.model(opt)
    model.train()

    # Init root path
    root_dir = os.path.join(opt.out_path, run_id)
    utils.cond_mkdir(root_dir)
    p.write_config_file(opt, [os.path.join(root_dir, 'config.txt')])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="PSNR_validation_epoch", dirpath=root_dir,
                                                       filename="model-{epoch:02d}-{PSNR_validation_epoch:.2f}",
                                                       save_top_k=1, mode="max", )

    # Init trainer
    trainer = Trainer(default_root_dir=root_dir, accelerator='gpu',
                      log_every_n_steps=400, gpus=1, max_epochs=opt.num_epochs, callbacks=[checkpoint_callback])

    # Fit Model
    trainer.fit(model, train_loader, val_loader)
    # Test Model
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()