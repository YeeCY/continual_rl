import torch
from torch.utils.data import ConcatDataset
import numpy as np
import tqdm

from src.mnist_cl.data import SubDataset, ExemplarDataset
from src.mnist_cl import utils
from src.mnist_cl.ewc_classifier import EwcClassifier
from src.mnist_cl.si_classifier import SiClassifier
from src.mnist_cl.agem_classifier import AgemClassifier


def train_cl(model, train_datasets, replay_mode="none", scenario="class", classes_per_task=None, iters=2000,
             batch_size=32, loss_cbs=None):
    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model.is_on_cuda()
    device = model.device()

    # Initiate possible sources for replay (no replay for 1st task)
    exact = False
    # previous_model = None

    # Register starting param-values (needed for "intelligent synapses").
    # if isinstance(model, SiClassifier):
    #     for n, p in model.named_parameters():
    #         if p.requires_grad:
    #             n = n.replace('.', '__')
    #             model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())

    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):
        # # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
        # if isinstance(model, ContinualLearner) and (model.si_c>0):
        #     W = {}
        #     p_old = {}
        #     for n, p in model.named_parameters():
        #         if p.requires_grad:
        #             n = n.replace('.', '__')
        #             W[n] = p.data.clone().zero_()
        #             p_old[n] = p.data.clone()

        # Find [active_classes]
        active_classes = None  # -> for Domain-IL scenario, always all classes are active
        if scenario == "task":
            # -for Task-IL scenario, create <list> with for all tasks so far a <list> with the active classes
            active_classes = [list(range(classes_per_task * i, classes_per_task * (i + 1))) for i in range(task)]
        elif scenario == "class":
            # -for Class-IL scenario, create one <list> with active classes of all tasks so far
            active_classes = list(range(classes_per_task * task))

        # Initialize # iters left on current data-loader(s)
        iters_left = iters_left_previous = 1
        if scenario == "task":
            iters_left_previous = [1] * (task - 1)
            data_loader_previous = [None] * (task - 1)

        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))

        # Loop over all iterations
        for batch_index in range(1, iters + 1):
            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left == 0:
                data_loader = iter(utils.get_data_loader(train_dataset, batch_size, cuda=cuda, drop_last=True))
                # NOTE:  [train_dataset]  is training-set of current task
                #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                iters_left = len(data_loader)
            if exact:
                if scenario == "task":
                    up_to_task = task - 1
                    batch_size_replay = int(np.ceil(batch_size/up_to_task)) if (up_to_task > 1) else batch_size
                    # -in Task-IL scenario, need separate replay for each task
                    for task_id in range(up_to_task):
                        batch_size_to_use = min(batch_size_replay, len(previous_datasets[task_id]))
                        iters_left_previous[task_id] -= 1
                        if iters_left_previous[task_id]==0:
                            data_loader_previous[task_id] = iter(utils.get_data_loader(
                                previous_datasets[task_id], batch_size_to_use, cuda=cuda, drop_last=True
                            ))
                            iters_left_previous[task_id] = len(data_loader_previous[task_id])
                else:
                    iters_left_previous -= 1
                    if iters_left_previous == 0:
                        batch_size_to_use = min(batch_size, len(ConcatDataset(previous_datasets)))
                        data_loader_previous = iter(utils.get_data_loader(ConcatDataset(previous_datasets),
                                                                          batch_size_to_use, cuda=cuda, drop_last=True))
                        iters_left_previous = len(data_loader_previous)


            # -----------------Collect data------------------#

            #####-----CURRENT BATCH-----#####
            x, y = next(data_loader)                                    #--> sample training data of current task
            y = y - classes_per_task*(task-1) if scenario == "task" else y  #--> ITL: adjust y-targets to 'active range'
            x, y = x.to(device), y.to(device)                           #--> transfer them to correct device

            #####-----REPLAYED BATCH-----#####
            if not exact:
                x_ = y_ = scores_ = None

            ##-->> Exact Replay <<--##
            if exact:
                scores_ = None
                if scenario in ("domain", "class"):
                    # Sample replayed training data, move to correct device
                    x_, y_ = next(data_loader_previous)
                    x_ = x_.to(device)
                    y_ = y_.to(device)
                elif scenario == "task":
                    # Sample replayed training data, wrap in (cuda-)Variables and store in lists
                    x_ = list()
                    y_ = list()
                    up_to_task = task if replay_mode=="offline" else task-1
                    for task_id in range(up_to_task):
                        x_temp, y_temp = next(data_loader_previous[task_id])
                        x_.append(x_temp.to(device))
                        y_temp = y_temp - (classes_per_task*task_id) #-> adjust y-targets to 'active range'
                        y_.append(y_temp.to(device))

            #---> Train MAIN MODEL
            if batch_index <= iters:
                # Train the main model with this batch
                if isinstance(model, EwcClassifier):
                    loss_dict = model.train_a_batch(x, y, active_classes=active_classes)
                elif isinstance(model, SiClassifier):
                    pass
                elif isinstance(model, AgemClassifier):
                    pass
                else:
                    raise RuntimeError(f"Unknown model type: {type(model)}")

                # Update running parameter importance estimates in W
                # if isinstance(model, ContinualLearner) and (model.si_c>0):
                #     for n, p in model.named_parameters():
                #         if p.requires_grad:
                #             n = n.replace('.', '__')
                #             if p.grad is not None:
                #                 W[n].add_(-p.grad*(p.detach()-p_old[n]))
                #             p_old[n] = p.detach().clone()

                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, task=task)

        ##----------> UPON FINISHING EACH TASK...

        # Close progres-bar(s)
        progress.close()

        # EWC: estimate Fisher Information matrix (FIM) and update term for quadratic penalty
        if isinstance(model, EwcClassifier):
            # -find allowed classes
            allowed_classes = list(
                range(classes_per_task*(task-1), classes_per_task*task)
            ) if scenario == "task" else (list(range(classes_per_task*task)) if scenario == "class" else None)
            # -estimate FI-matrix
            model.estimate_fisher(train_dataset, allowed_classes=allowed_classes)

        # SI: calculate and update the normalized path integral
        if isinstance(model, SiClassifier):
            # TODO (chongyi zheng)
            model.update_omegas()

        # EXEMPLARS: update exemplar sets
        if replay_mode == "exemplars":
            exemplars_per_class = int(np.floor(model.memory_budget / (classes_per_task*task)))
            # reduce examplar-sets
            model.reduce_exemplar_sets(exemplars_per_class)
            # for each new class trained on, construct examplar-set
            new_classes = list(range(classes_per_task)) if scenario == "domain" else \
                list(range(classes_per_task * (task - 1), classes_per_task * task))
            for class_id in new_classes:
                # create new dataset containing only all examples of this class
                class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                # based on this dataset, construct new exemplar-set for this class
                model.construct_exemplar_set(dataset=class_dataset, n=exemplars_per_class)
            model.compute_means = True

        # REPLAY: update source for replay
        # previous_model = copy.deepcopy(model).eval()
        if replay_mode in ('exemplars', 'exact'):
            exact = True
            if replay_mode == "exact":
                previous_datasets = train_datasets[:task]
            else:
                if scenario == "task":
                    previous_datasets = []
                    for task_id in range(task):
                        previous_datasets.append(
                            ExemplarDataset(
                                model.exemplar_sets[
                                (classes_per_task * task_id):(classes_per_task * (task_id + 1))],
                                target_transform=lambda y, x=classes_per_task * task_id: y + x)
                        )
                else:
                    target_transform = (lambda y, x=classes_per_task: y % x) if scenario == "domain" else None
                    previous_datasets = [
                        ExemplarDataset(model.exemplar_sets, target_transform=target_transform)]
