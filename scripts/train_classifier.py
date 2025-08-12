import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import ubcrl.common.utils as utils
from ubcrl.classify.classifier import mujoco_safety_gymnasium_dict, TrajNPData, MujocoNPDataset, PtEstGRU, DistributionGRU, collate, TrajHDF5Data


def main(env, list_npz_paths, hdf5_file, modeldir,
         # storedatasetdir,
         logdir, testsplit, gruunits, grulayers, dropout, batchsize,
         # target_loss,
         target_accuracy, learn_rate, distribution_model, labeling_noise, seed, deviceno):

    if env in mujoco_safety_gymnasium_dict:
        feature_dim = mujoco_safety_gymnasium_dict[env]['state_dim'] + mujoco_safety_gymnasium_dict[env]['action_dim']
    else:
        print("Given env not recognized")
        sys.exit(1)

    assert (list_npz_paths is not None and hdf5_file is None) or (list_npz_paths is None and hdf5_file is not None), \
        "Either NPZ file paths or HDF5 file location must be specified"

    torch.manual_seed(seed)
    np.random.seed(seed)

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(deviceno)

    utils.set_device(deviceno)

    # creating a new directory
    Path(modeldir).mkdir(parents=True, exist_ok=True)
    Path(logdir).mkdir(parents=True, exist_ok=True)

    datetime_str = datetime.now().strftime("%d-%m-%Y_%H%M%S")
    logfilename = (logdir + "/train" + ("_Distribution" if distribution_model else "_") + "GRU_"
                   + env + "_" + datetime_str + ".log")

    logging.basicConfig(filename=logfilename,
                        format='%(asctime)s %(message)s',
                        filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    today_str = datetime.now().strftime("%d-%m-%Y")
    Path(os.path.join(modeldir, today_str)).mkdir(parents=True, exist_ok=True)
    model_file_prefix = os.path.join(modeldir, today_str, env
                                     + ('_Distribution' if distribution_model else '_') +  'GRU_'
                                     + str(gruunits) + '_' + str(grulayers) + '_' + str(batchsize))
    dataset_file_prefix = os.path.join(modeldir, today_str, env)

    # obs, action, reward, cost, next_obs, done
    logger.info("Loading trajectories from data files...")
    # print(datetime.now().strftime("%d-%m-%Y_%H%M%S") + " Before splitting NPZ data",
    #       flush=True)

    if list_npz_paths is not None:
        trajectories_data = TrajNPData(list_npz_paths, mujoco_safety_gymnasium_dict[env]['horizon'])
    else:
        trajectories_data = TrajHDF5Data(hdf5_file, mujoco_safety_gymnasium_dict[env]['horizon'],
                                         mujoco_safety_gymnasium_dict[env]['state_dim'], mujoco_safety_gymnasium_dict[env]['action_dim'])

    all_idx = np.arange(trajectories_data.get_num_traj())
    np.random.shuffle(all_idx)
    split_idx = int(len(all_idx) * testsplit)
    train_idx, test_idx = all_idx[split_idx:], all_idx[:split_idx]
    # print(datetime.now().strftime("%d-%m-%Y_%H%M%S") + " After splitting NPZ data",
    #       flush=True)
    logger.info("Completed loading of trajectories...")

    logger.info("Loading train dataset...")
    train_dataset = MujocoNPDataset(mujoco_domain=env, np_data=trajectories_data, indices=train_idx, noise=labeling_noise)
    torch.save(train_dataset, str(dataset_file_prefix) + '_traindataset.pt')
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize,
                                  collate_fn=collate, shuffle=True)
    logger.info("Completed loading of train dataset...")

    logger.info("Loading test dataset...")
    test_dataset = MujocoNPDataset(mujoco_domain=env, np_data=trajectories_data, indices=test_idx, noise=labeling_noise)
    torch.save(test_dataset, str(dataset_file_prefix) + '_testdataset.pt')
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize,
                                 collate_fn=collate, shuffle=True)
    logger.info("Completed loading of test dataset...")

    if distribution_model:
        net = DistributionGRU(feature_dim=feature_dim, nb_gru_units=gruunits, batch_size=batchsize,
                              gru_layers=grulayers, dropout=dropout, mlp_arch=mujoco_safety_gymnasium_dict[env]['decoder_arch'],
                              loc_offset=mujoco_safety_gymnasium_dict[env]['loc_offset'],
                              log_std_offset=mujoco_safety_gymnasium_dict[env]['log_std_offset']).to(utils.device)
    else:
        net = PtEstGRU(feature_dim=feature_dim, nb_gru_units=gruunits, batch_size=batchsize,
                       gru_layers=grulayers, dropout=dropout,
                       mlp_arch=mujoco_safety_gymnasium_dict[env]['decoder_arch']).to(utils.device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)  # , momentum=0.9)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learn_rate, weight_decay=1e-4)
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)


    logger.info("Training classifier, lr = " + str(learn_rate))

    epoch = 0
    ave_valid_loss, valid_accuracy = float('Inf'), float('-Inf')
    # for epoch in range(num_epochs):  # loop over the dataset multiple times
    # while ave_valid_loss > target_loss:
    while valid_accuracy < target_accuracy:
        net.train()
        running_loss, running_loss_train = 0.0, 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, input_lengths = data
            labels = labels.reshape(-1, 1)

            # forward + backward + optimize
            outputs, dict_log_c_out, h_out, _ = net(inputs, input_lengths)
            # print(outputs.shape)
            # print(log_c_out.shape)
            # loss = net.loss(outputs, labels.float().reshape(-1, 1))
            loss = net.loss(outputs, labels.float())

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss_train += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                logger.info('[%d, %5d] loss: %.4f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        logger.info('[%d] Training loss: %.4f' % (epoch + 1, running_loss_train / len(train_dataloader)))

        net.eval()

        running_valid_loss, valid_correct = 0.0, 0
        valid_tp, valid_tn, valid_fp, valid_fn = 0, 0, 0, 0

        for j, valid_data in enumerate(test_dataloader, 0):
            inputs_valid, labels_valid, input_lengths_valid = valid_data
            labels_valid = labels_valid.reshape(-1, 1)
            # labels_valid = labels_valid.float().reshape(-1, 1)
            valid_outputs, dict_valid_log_c_out, valid_h_out, _ = net(inputs_valid, input_lengths_valid)
            valid_loss = net.loss(valid_outputs, labels_valid.float())

            running_valid_loss += valid_loss.item()

            labels_valid = labels_valid.bool()
            valid_correct += ((valid_outputs > 0.5) == labels_valid).sum().item()
            valid_tp += ((valid_outputs > 0.5) & labels_valid).sum().item()
            valid_fp += ((valid_outputs > 0.5) & labels_valid.logical_not()).sum().item()
            valid_tn += ((valid_outputs <= 0.5) & labels_valid.logical_not()).sum().item()
            valid_fn += ((valid_outputs <= 0.5) & labels_valid).sum().item()

        ave_valid_loss = running_valid_loss / len(test_dataloader)
        valid_accuracy = valid_correct / len(test_dataloader.dataset)

        logger.info('[%d] Validation loss: %.4f' % (epoch + 1, ave_valid_loss))
        logger.info(f"Validation Error: \n Accuracy: {(100 * valid_accuracy):>0.1f}%, Avg loss: {ave_valid_loss:>8f} \n")
        logger.info(f"Validation Error: \n Precision: {(100 * valid_tp / (valid_tp + valid_fp)):>0.1f}% \n")
        logger.info(f"Validation Error: \n Recall: {(100 * valid_tp / (valid_tp + valid_fn)):>0.1f}% \n")
        epoch += 1

        # torch.save(net.state_dict(),
        #            str(model_file_prefix) + '_' + str(epoch) + '.pt')

    logger.info("Completed training of classifier...")

    torch.save(net.state_dict(),
               str(model_file_prefix) + '.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, required=True,
                        help='Environment name: select from [SafetyHalfCheetahVelocity-v1, SafetyHopperVelocity-v1, SafetyWalker2dVelocity-v1, SafetyAntVelocity-v1, SafetySwimmerVelocity-v1]')
    parser.add_argument('--list_npzfilepaths', nargs='*', type=str,  # required=True,
                        help='Paths where the NPZ trajectory datafiles are stored')
    parser.add_argument('--hdf5_file', type=str,  # required=True,
                        help='Path where the HDF5 trajectory file is stored')
    parser.add_argument('-modeldir', type=str, default='model',
                        help='Directory to which models will be saved (default: ./model)')
    # parser.add_argument('-storedatasetdir', type=str, default='dataset',
    #                     help='Directory where the train and test datasets saved (default: ./dataset)')
    parser.add_argument('-logdir', type=str, default='log',
                        help='Directory to which results will be logged (default: ./log)'
                        )
    parser.add_argument('-testsplit', type=float, default=0.1,
                        help='Split proportion for test dataset (default: 0.1')
    parser.add_argument('-gruunits', type=int, default=8,
                        help='Number of GRU units (default: 8)'
                        )
    parser.add_argument('-grulayers', type=int, default=2,
                        help='Number of GRU layers (default: 2)'
                        )
    parser.add_argument('-dropout', type=float, default=0.0,
                        help='Dropout of Stacked GRU (default: 0.0)'
                        )
    parser.add_argument('-batchsize', type=int, default=256,
                        help='Batch size (default: 256)'
                        )
    # parser.add_argument('-epoch', type=int, default=10,
    #                     help='Number of training epochs (default: 10)'
    #                     )
    # parser.add_argument('-targetloss', type=float, default=0.015,
    #                     help='Target loss to stop training (default: 0.015)'
    #                     )
    parser.add_argument('-targetacc', type=float, default=0.95,
                        help='Target accuracy to stop training (default: 0.95)'
                        )
    parser.add_argument('-lr', '--learn_rate', type=float, default=0.001,
                        help='Classifier learning rate (default: 0.001)'
                        )
    parser.add_argument('-distribution_model', action='store_true',
                        help='Use the distribution model'
                        )
    parser.add_argument('-labeling_noise', type=float, default=0.0,
                        help='Labeling noise (in proportion) while labeling trajectory (default: 0.0)'
                        )
    parser.add_argument('-seed', type=int, default=999,
                        help='Random seed to be used (default: 999)'
                        )
    parser.add_argument('-deviceno', type=int, default=0,
                        help='GPU device number to use (default: 0)'
                        )
    args = parser.parse_args()

    main(args.env, args.list_npzfilepaths, args.hdf5_file, args.modeldir,
         # args.storedatasetdir,
         args.logdir, args.testsplit, args.gruunits, args.grulayers, args.dropout, args.batchsize,
         # args.targetloss,
         args.targetacc, args.learn_rate, args.distribution_model, args.labeling_noise, args.seed, args.deviceno)
