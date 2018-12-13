import os
import logging
import fitsio

logger = logging.getLogger(__name__)

def get_run_dir(run):
    """
    get the base run dir
    """
    bdir=os.environ['FITCOSMOS_DIR']
    return os.path.join(
        bdir,
        run,
    )

def get_collated_dir(run):
    """
    get the collated directory
    """
    run_dir=get_run_dir(run)
    return os.path.join(
        run_dir,
        'collated',
    )

def get_split_dir(run):
    """
    get the split output directory
    """
    run_dir=get_run_dir(run)
    return os.path.join(
        run_dir,
        'splits',
    )

def get_script_dir(run):
    """
    directory for scripts
    """
    run_dir=get_run_dir(run)
    return os.path.join(
        run_dir,
        'scripts',
    )

def get_script_path(run, isplit):
    """
    directory for scripts
    """
    script_dir=get_script_dir(run)

    fname='%s-%06d.sh' % (run, isplit)
    return os.path.join(
        script_dir,
        fname,
    )




def load_fofs(fof_filename):
    """
    load FoF information from the file
    """
    logger.info('loading fofs: %s' % fof_filename)
    with fitsio.FITS(fof_filename
                    ) as fits:
        nbrs=fits['nbrs'][:]
        fofs=fits['fofs'][:]

    return nbrs, fofs


