"""
TODO:

    - make MEDS box size stuff pre-calculated in arcsec,
      with typical DES psf taken into account (e.g. FWHM=1.2)
    - make guesses in arcsec too (put guesser into this code don't use
      default one in mof)
    - make sure good guesses being used.
    - do output
    - docs

"""
import numpy as np
import logging
import ngmix
import ngmix.medsreaders
import fitsio
import yaml
import esutil as eu

from . import fitting

logger = logging.getLogger(__name__)

class Processor(object):
    def __init__(self, args):
        self.args=args

        self._set_rng()
        self._load_conf()
        self._load_meds_files()
        self._load_fofs()
        self._set_fof_range()
        self._set_fitter()

    def go(self):

        olist=[]
        for fofid in range(self.start,self.end+1):
            logger.info('processing: %d:%d' % (fofid,self.end))
            output = self._process_fof(fofid)
            olist.append(output)

        output = eu.numpy_util.combine_arrlist(olist)

        self._write_output(output)

    def _write_output(self, output):
        logger.info('writing output: %s' % self.args.output)
        with fitsio.FITS(self.args.output,'rw',clobber=True) as fits:
            fits.write(output)

    def _process_fof(self, fofid):
        w,=np.where(self.fofs['fofid'] == fofid)
        logger.debug('%d objects' % w.size)

        indices=self.fofs['number'][w]-1

        mbobs_list=[]
        for index in indices:
            mbobs=self.mb_meds.get_mbobs(
                index,
                weight_type='weight',
            )
            for band,obslist in enumerate(mbobs):
                m=self.mb_meds.mlist[band]
                meta = {
                    #'Tsky': 2*m['flux_radius']**2,
                    'T': 2* (m['iso_radius'][index]*0.5)**2,
                    'flux': m['flux'][index],
                }

                obslist.meta.update(meta)

            mbobs_list.append( mbobs )

        output = self.fitter.go(mbobs_list)
        output['fof_id'] = fofid
        return output

    def _set_rng(self):
        self.rng = np.random.RandomState(self.args.seed)

    def _load_conf(self):
        logger.info('loading config: %s' % self.args.config)
        with open(self.args.config) as fobj:
            self.config = yaml.load(fobj)

    def _set_fitter(self):
        self.fitter = fitting.MOFFitter(
            self.config,
            self.mb_meds.nband,
            self.rng,
        ) 

    def _load_fofs(self):
        logger.info('loading fofs: %s' % self.args.fofs)
        with fitsio.FITS(self.args.fofs) as fits:
            #self.nbrs=fits['nbrs'][:]
            self.fofs=fits['fofs'][:]

    def _set_fof_range(self):
        self.start=self.args.start
        self.end=self.args.end

        if self.start is None:
            self.start = 0
        if self.end is None:
            self.end = self.mb_meds.mlist[0].size-1

    def _load_meds_files(self):
        mlist=[]
        for f in self.args.meds:
            logger.info('loading meds: %s' % f)
            mlist.append( ngmix.medsreaders.NGMixMEDS(f) )

        self.mb_meds = ngmix.medsreaders.MultiBandNGMixMEDS(
            mlist,
        )
