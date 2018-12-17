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

        self['fof_file'] = os.path.abspath(args.fofs)
        self['fit_config'] = os.path.abspath(args.fit_config)
        meds_files = [
            os.path.abspath(mf) for mf in self.args.meds
        ]
        self.meds_files = ' '.join(meds_files)

        self._load_config()
        self._set_rng()
        self._load_fofs()
        self._make_dirs()

    def go(self):
        num_fofs = self.fofs['fofid'].max()
        fof_splits = split.get_splits(num_fofs, self['chunksize'])

        for isplit,fof_split in enumerate(fof_splits):
            logger.info('%s %s' % (isplit,fof_split))
            self._write_split(isplit, fof_split)

    def _make_dirs(self):
        dir=files.get_split_dir(self['run'])
        try:
            os.makedirs(dir)
        except:
            pass

        dir=files.get_script_dir(self['run'])
        try:
            os.makedirs(dir)
        except:
            pass


    def _write_split(self,isplit,fof_split):
        raise NotImplementedError('implement in child class')

    def _write_script(self, isplit, fof_split):
        start, end = fof_split
        fname=files.get_script_path(self['run'], start, end)
        logger.info('script: %s' % fname)


        output_file = files.get_split_output(
            self['run'],
            start,
            end,
            ext='fits',
        )
        log_file = files.get_split_output(
            self['run'],
            start,
            end,
            ext='log',
        )

        d={}
        d['seed'] = self._get_seed()
        d['output_file'] = os.path.abspath(output_file)
        d['fit_config'] = self['fit_config']
        d['fof_file'] = self['fof_file']
        d['start'] = start
        d['end'] = end
        d['meds_files'] = self.meds_files
        d['logfile'] = os.path.abspath(log_file)

        text=_script_template % d

        with open(fname,'w') as fobj:
            fobj.write(text)

        os.system('chmod 755 %s' % fname)

    def _get_seed(self):
        return self.rng.randint(0,2**31)

    def _load_config(self):
        with open(self.args.run_config) as fobj:
            run_config=yaml.load(fobj)
        self.update(run_config)

        bname=os.path.basename(self.args.run_config)
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



seed="%(seed)s"
output="%(output_file)s"
config="%(fit_config)s"
fofs="%(fof_file)s"
start="%(start)d"
end="%(end)d"
meds="%(meds_files)s"
logfile="%(logfile)s"

logbase=$(basename $logfile)
tmplog=$tmpdir/$logbase

fitcosmos \
    --seed=$seed \
    --config=$config \
    --output=$output \
    --fofs=$fofs \
    --start=$start \
    --end=$end \
    $meds &> $tmplog

mv -vf $tmplog $logfile
"""


