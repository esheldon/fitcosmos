"""
TODO

    - condor submit with master script rather than individual scripts?
    - wq probably makes more sense but it is totally booked up
    - batch maker just takes meds run and conf and figures
    out the rest?
    - make it write a fof making script
      note it uses multiple cores via numba+mkl
"""
import os
import numpy as np
import yaml
import logging
import fitsio
from . import split
from . import files
from .files import StagedOutFile

logger = logging.getLogger(__name__)

class BatchBase(dict):
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
        """
        write for all FoF groups
        """
        num_fofs = self.fofs['fofid'].max()
        fof_splits = split.get_splits(num_fofs, self['chunksize'])

        for isplit,fof_split in enumerate(fof_splits):
            logger.info('%s %s' % (isplit,fof_split))
            self._write_split(isplit, fof_split)

    def _make_dirs(self):
        dirs = [
            files.get_split_dir(self['run']),
            files.get_script_dir(self['run']),
            files.get_collated_dir(self['run']),
        ]
        for d in dirs:
            try:
                os.makedirs(d)
            except:
                pass

    def _write_split(self,isplit,fof_split):
        raise NotImplementedError('implement in child class')

    def _write_script(self, isplit, fof_split):
        start, end = fof_split
        fname=files.get_script_path(self['run'], start, end)

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

        if self.args.missing and os.path.exists(output_file):
            if os.path.exists(fname):
                os.remove(fname)
            return

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

        logger.info('script: %s' % fname)
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

class ShellBatch(BatchBase):
    """
    just write out the scripts, no submit files
    """
    def _write_split(self, isplit, fof_split):
        self._write_script(isplit, fof_split)

class WQBatch(ShellBatch):
    """
    just write out the scripts, no submit files
    """
    def _write_split(self, isplit, fof_split):
        super(WQBatch,self)._write_split(isplit, fof_split)
        self._write_wq_script(isplit, fof_split)

    def _write_wq_script(self, isplit, fof_split):
        """
        write the wq submit script
        """
        start, end = fof_split

        script_file=files.get_script_path(self['run'], start, end)
        wq_file=files.get_wq_path(self['run'], start, end)
        job_name='%s-%06d-%06d' % (self['run'], start, end)

        output_file = files.get_split_output(
            self['run'],
            start,
            end,
            ext='fits',
        )

        if self.args.missing and os.path.exists(output_file):
            if os.path.exists(wq_file):
                os.remove(wq_file)
            return

        logger.info('wq script: %s' % wq_file)

        d={}
        d['script'] = script_file
        d['job_name'] = job_name

        text=_wq_template % d

        with open(wq_file,'w') as fobj:
            fobj.write(text)

class CondorBatch(BatchBase):
    """
    just write out the scripts, no submit files
    """

    def go(self):
        """
        write all the scripts
        """

        self._write_master()
        num_fofs = self.fofs['fofid'].max()
        fof_splits = split.get_splits(num_fofs, self['chunksize'])

        njobs=0
        fobj=None

        icondor=0
        for isplit,fof_split in enumerate(fof_splits):
            if njobs % self['jobs_per_sub']==0:
                if fobj is not None:
                    fobj.close()
                fobj = self._open_condor_script(icondor)
                icondor += 1

            self._write_split(fobj, isplit, fof_split)

            njobs += 1

    def _write_split(self, fobj, isplit, fof_split):
        """
        write the lines to the submit file object
        """

        start, end = fof_split

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
        d['logfile'] = os.path.abspath(log_file)
        d['job_name']='%s-%06d-%06d' % (self['run'], start, end)

        job = _condor_job_template % d

        fobj.write(job)

    def _make_dirs(self):
        dirs = [
            files.get_split_dir(self['run']),
            files.get_condor_dir(self['run']),
            files.get_collated_dir(self['run']),
        ]
        for d in dirs:
            try:
                os.makedirs(d)
            except:
                pass

    def _write_master(self):
        """
        write the master script
        """
        text = _condor_master_template % {
            'meds_files':self.meds_files,
        }
        master_script=files.get_condor_master_path(self['run'])
        print('writing master:',master_script)
        with open(master_script,'w') as fobj:
            fobj.write(text)

        os.system('chmod 755 %s' % master_script)

    def _open_condor_script(self, icondor):
        """
        open the condor script
        """

        fname=files.get_condor_script(self['run'], icondor)
        print('condor script:',fname)
        fobj = open(fname,'w')

        master_script=files.get_condor_master_path(self['run'])
        text = _condor_head % {
            'master_script':master_script,
        }
        fobj.write(text)

        return fobj



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


_wq_template=r"""
command: |
    . ~/.bashrc
    source activate cosmos
    bash %(script)s

job_name: %(job_name)s
"""

_condor_head = r"""
Universe        = vanilla

Notification    = Never

# Run this exe with these args
Executable      = %(master_script)s

Image_Size       =  1000000

GetEnv = True

kill_sig        = SIGINT

#requirements = (cpu_experiment == "sdcc")

+Experiment     = "astro"
"""

_condor_job_template="""
+job_name = "%(job_name)s"
Arguments = %(seed)d %(output_file)s %(fit_config)s %(fof_file)s %(start)d %(end)d %(logfile)s

Queue
"""


_condor_master_template=r"""#!/bin/bash

if [[ -n $_CONDOR_SCRATCH_DIR ]]; then
    tmpdir=$_CONDOR_SCRATCH_DIR
    export TMPDIR=$tmpdir
else
    tmpdir=$TMPDIR
    mkdir -p $tmpdir
fi

export OMP_NUM_THREADS=1

seed="$1"
output="$2"
config="$3"
fofs="$4"
start="$5"
end="$6"
logfile="$7"

logbase=$(basename $logfile)
tmplog=$tmpdir/$logbase

meds="%(meds_files)s"

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


