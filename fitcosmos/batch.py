import os
import numpy as np
import yaml
import logging
from . import split
from . import files

logger = logging.getLogger(__name__)

class BatchMakerBase(dict):
    def __init__(self, args):
        self.args=args

        self._load_config()
        self._set_rng()
        self._load_fofs()

    def go(self):
        num_fofs = self.fofs['fofid'].max()
        fof_splits = split.get_splits(num_fofs, self['chunksize'])

        for isplit,fof_split in enumerate(fof_splits):
            logger.info('%s %s' % (isplit,fof_split))
            self._write_split(isplit, fof_split)

    def _write_split(self,isplit,fof_split):
        raise NotImplementedError('implement in child class')

    def _write_script(self, isplit, fof_split):
        fname=files.get_script_path(self['run'], isplit)
        logger.info('script: %s' % fname)

    def _load_config(self):
        with open(self.args.config) as fobj:
            run_config=yaml.load(fobj)
        self.update(run_config)

        bname=os.path.basename(self.args.config)
        self['run'] = bname.replace('.yaml','')

    def _load_fofs(self):
        nbrs,fofs=files.load_fofs(self.args.fofs)
        self.fofs=fofs


    def _set_rng(self):
        self.rng = np.random.RandomState(self['seed'])

class ShellBatchMaker(BatchMakerBase):
    """
    just write out the scripts, no submit files
    """
    def _write_split(self, isplit, fof_split):
        self._write_script(isplit, fof_split)

_script_template=r"""#!/bin/bash

if [[ -n $_CONDOR_SCRATCH_DIR ]]; then
    # the condor system creates a scratch directory for us,
    # and cleans up afterward
    tmpdir=$_CONDOR_SCRATCH_DIR
    export TMPDIR=$tmpdir
else
    # otherwise use the TMPDIR
    tmpdir=$TMPDIR
    mkdir -p $tmpdir
fi

export OMP_NUM_THREADS=1


logbase=$(basename $logfile)
tmplog=$tmpdir/$logbase

seed="{seed}"
output="{output_file}"
config="{fit_config}"
fofs="{fof_file}"
start="{start}"
end="{end}"
meds="{meds_files}"
logfile="{logfile}"

fitcosmos \
    --seed=$seed
    --output=$output \
    --config=$config \
    --fofs=$fofs \
    --start=$start \
    --end=$end
    $meds &> $tmplog

mv -vf $tmplog $logfile
"""


