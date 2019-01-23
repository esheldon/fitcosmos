import os
import shutil
import logging
import fitsio

logger = logging.getLogger(__name__)

def get_tempdir():
    return os.environ['TMPDIR']

def get_run_dir(run):
    """
    get the base run dir
    """
    bdir=os.environ['FITCOSMOS_DIR']
    return os.path.join(
        bdir,
        run,
    )

def get_fof_dir(run):
    """
    get the directory holding fofs
    """
    run_dir=get_run_dir(run)
    return os.path.join(
        run_dir,
        'fofs',
    )

def get_fof_file(run):
    """
    get the directory holding fofs
    """
    fof_dir=get_fof_dir(run)
    fname='%s-fofs.fits' % run
    return os.path.join(
        fof_dir,
        fname,
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

def get_collated_file(run):
    """
    get the collated file name
    """
    split_dir=get_collated_dir(run)
    fname = '%s.fits' % run
    return os.path.join(
        split_dir,
        fname,
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

def get_split_output(run, start, end, ext='fits'):
    """
    get the split output file
    """
    split_dir=get_split_dir(run)
    fname = '%s-%06d-%06d.%s' % (run,start, end, ext)
    return os.path.join(
        split_dir,
        fname,
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

def get_script_path(run, start, end):
    """
    directory for scripts
    """
    script_dir=get_script_dir(run)

    fname = '%s-%06d-%06d.sh' % (run, start, end)
    return os.path.join(
        script_dir,
        fname,
    )

def get_fof_script_path(run):
    """
    directory for scripts
    """
    script_dir=get_script_dir(run)

    fname = '%s-make-fofs.sh' % run
    return os.path.join(
        script_dir,
        fname,
    )


def get_wq_path(run, start, end):
    """
    directory for scripts
    """
    script_dir=get_script_dir(run)

    fname = '%s-%06d-%06d.yaml' % (run, start, end)
    return os.path.join(
        script_dir,
        fname,
    )


def get_condor_dir(run):
    """
    directory for scripts
    """
    run_dir=get_run_dir(run)
    return os.path.join(
        run_dir,
        'condor',
    )

def get_condor_master_path(run):
    """
    master script for condor
    """
    condor_dir=get_condor_dir(run)

    fname = '%s-master.sh' % run
    return os.path.join(
        condor_dir,
        fname,
    )

def get_condor_script(run, icondor):
    """
    submit script
    """
    condor_dir=get_condor_dir(run)

    fname = '%s-%06d.condor' % (run, icondor)
    return os.path.join(
        condor_dir,
        fname,
    )




def load_fofs(fof_filename):
    """
    load FoF information from the file
    """
    logger.info('loading fofs: %s' % fof_filename)
    with fitsio.FITS(fof_filename) as fits:
        nbrs=fits['nbrs'][:]
        fofs=fits['fofs'][:]

    return nbrs, fofs

class StagedOutFile(object):
    """
    A class to represent a staged file
    If tmpdir=None no staging is performed and the original file
    path is used
    parameters
    ----------
    fname: string
        Final destination path for file
    tmpdir: string, optional
        If not sent, or None, the final path is used and no staging
        is performed
    must_exist: bool, optional
        If True, the file to be staged must exist at the time of staging
        or an IOError is thrown. If False, this is silently ignored.
        Default False.
    examples
    --------

    fname="/home/jill/output.dat"
    tmpdir="/tmp"
    with StagedOutFile(fname,tmpdir=tmpdir) as sf:
        with open(sf.path,'w') as fobj:
            fobj.write("some data")

    """
    def __init__(self, fname, tmpdir=None, must_exist=False):

        self.must_exist = must_exist
        self.was_staged_out = False

        self._set_paths(fname, tmpdir=tmpdir)


    def _set_paths(self, fname, tmpdir=None):
        fname=expandpath(fname)

        self.final_path = fname

        if tmpdir is not None:
            self.tmpdir = expandpath(tmpdir)
        else:
            self.tmpdir = tmpdir

        fdir = os.path.dirname(self.final_path)

        if self.tmpdir is None:
            self.is_temp = False
            self.path = self.final_path
        else:
            if not os.path.exists(self.tmpdir):
                os.makedirs(self.tmpdir)

            bname = os.path.basename(fname)
            self.path = os.path.join(self.tmpdir, bname)

            if self.tmpdir==fdir:
                # the user sent tmpdir as the final output dir, no
                # staging is performed
                self.is_temp = False
            else:
                self.is_temp = True

    def stage_out(self):
        """
        if a tempdir was used, move the file to its final destination
        note you normally would not call this yourself, but rather use a
        context, in which case this method is called for you
        with StagedOutFile(fname,tmpdir=tmpdir) as sf:
            #do something
        """

        if self.is_temp and not self.was_staged_out:
            if not os.path.exists(self.path):
                if self.must_exist:
                    mess = "temporary file not found: %s" % self.path
                    raise IOError(mess)
                else:
                    return

            if os.path.exists(self.final_path):
                print("removing existing file:",self.final_path)
                os.remove(self.final_path)

            makedir_fromfile(self.final_path)

            print("staging out '%s' -> '%s'" % (self.path,self.final_path))
            shutil.move(self.path,self.final_path)

        self.was_staged_out=True

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.stage_out()

def expandpath(path):
    """
    expand environment variables, user home directories (~), and convert
    to an absolute path
    """
    path=os.path.expandvars(path)
    path=os.path.expanduser(path)
    path=os.path.realpath(path)
    return path


def makedir_fromfile(fname):
    """
    extract the directory and make it if it does not exist
    """
    dname=os.path.dirname(fname)
    try_makedir(dname)

def try_makedir(dir):
    """
    try to make the directory
    """
    if not os.path.exists(dir):
        try:
            print("making directory:",dir)
            os.makedirs(dir)
        except:
            # probably a race condition
            pass


