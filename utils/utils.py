def get_name(opts):
    name = '{}_{}'.format(opts.data_name, opts.model_name)
    if opts.fgr_lamb > 0:
        name = name + '_fgr_{}'.format(opts.fgr_lamb)
        if opts.rs_sigma > 0:
            name = name + '_rs_{}'.format(opts.rs_sigma)
    return name
