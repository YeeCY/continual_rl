import data


def get_param_stamp_from_args(args):
    '''To get param-stamp a bit quicker.'''
    from encoder import Classifier
    from vae_models import AutoEncoder

    scenario = args.scenario
    # If Task-IL scenario is chosen with single-headed output layer, set args.scenario to "domain"
    # (but note that when XdG is used, task-identity information is being used so the actual scenario is still Task-IL)
    if args.singlehead and args.scenario=="task":
        scenario="domain"

    config = data.get_multitask_experiment(
        name=args.experiment, scenario=scenario, tasks=args.tasks, data_dir=args.d_dir, only_config=True,
        verbose=False,
    )

    if args.feedback:
        model = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            fc_layers=args.fc_lay, fc_units=args.fc_units, z_dim=args.z_dim,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl,
        )
        model.lamda_pl = 1.
    else:
        model = Classifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            fc_layers=args.fc_lay, fc_units=args.fc_units, fc_drop=args.fc_drop, fc_nl=args.fc_nl,
            fc_bn=True if args.fc_bn=="yes" else False, excit_buffer=True if args.xdg and args.gating_prop>0 else False,
        )

    train_gen = True if (args.replay=="generative" and not args.feedback) else False
    if train_gen:
        generator = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'],
            fc_layers=args.g_fc_lay, fc_units=args.g_fc_uni, z_dim=args.g_z_dim, classes=config['classes'],
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn == "yes" else False, fc_nl=args.fc_nl,
        )

    model_name = model.name
    replay_model_name = generator.name if train_gen else None
    param_stamp = get_param_stamp(args, model_name, verbose=False, replay=False if (args.replay=="none") else True,
                                  replay_model_name=replay_model_name)
    return param_stamp



def get_param_stamp(args, verbose=True, replay=False, replay_model_name=None):
    '''Based on the input-arguments, produce a "parameter-stamp".'''

    # -for task
    multi_n_stamp = "{n}-{set}".format(n=args.num_tasks, set=args.scenario) if hasattr(args, "tasks") else ""
    task_stamp = "{exp}{multi_n}".format(exp=args.dataset, multi_n=multi_n_stamp)
    if verbose:
        print(" --> task:          "+task_stamp)

    # -for model
    if args.ewc:
        model_stamp = 'ewc'
    elif args.si:
        model_stamp = 'si'
    elif args.agem:
        model_stamp = 'agem'
    elif args.cmaml:
        model_stamp = 'cmaml'
    else:
        raise RuntimeError("Unknown algorithm")
    if verbose:
        print(" --> model:         "+model_stamp)

    # -for hyper-parameters
    hyper_stamp = "{i_e}{num}-lr{lr}-b{bsz}".format(
        i_e="e" if args.iters is None else "i", num=args.epochs if args.iters is None else args.iters, lr=args.lr,
        bsz=args.batch_size,
    )
    if verbose:
        print(" --> hyper-params:  " + hyper_stamp)

    # -for EWC / SI
    if hasattr(args, 'ewc') and ((args.ewc and args.ewc_lambda > 0) or (args.si and args.si_c > 0)):
        ewc_stamp = "EWC{l}-{fi}{o}".format(
            l=args.ewc_lambda,
            fi="{}".format("N" if args.ewc_fisher_sample_size is None else args.fisher_n),
            o="-O{}".format(args.ewc_gamma) if args.ewc_online else "",
        ) if (args.ewc and args.ewc_lambda > 0) else ""
        si_stamp = "SI{c}-{eps}".format(c=args.si_c, eps=args.si_epsilon) if (args.si and args.si_c > 0) else ""
        both = "--" if (args.ewc and args.ewc_lambda > 0) and (args.si and args.si_c > 0) else ""
        if verbose and args.ewc and args.ewc_lambda > 0:
            print(" --> EWC:           " + ewc_stamp)
        if verbose and args.si and args.si_c > 0:
            print(" --> SI:            " + si_stamp)
    ewc_stamp = "--{}{}{}".format(ewc_stamp, both, si_stamp) if (
            hasattr(args, 'ewc') and ((args.ewc and args.ewc_lambda > 0) or (args.si and args.si_c > 0))
    ) else ""

    # -for replay
    if replay:
        replay_stamp = "{rep}{agem}{model}".format(
            rep=args.replay,
            agem="-aGEM" if args.agem else "",
            model="" if (replay_model_name is None) else "-{}".format(replay_model_name),
        )
        if verbose:
            print(" --> replay:        " + replay_stamp)
    replay_stamp = "--{}".format(replay_stamp) if replay else ""


    # --> combine
    param_stamp = "{}--{}--{}{}{}{}".format(
        task_stamp, model_stamp, hyper_stamp, ewc_stamp, replay_stamp,
        "-s{}".format(args.seed) if not args.seed == 0 else "",
    )

    ## Print param-stamp on screen and return
    if verbose:
        print(param_stamp)
    return param_stamp
