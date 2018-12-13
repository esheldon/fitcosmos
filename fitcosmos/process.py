"""
TODO:

    - allow limiting to some max postage stamp size
        - this means removing the entire group from fitting though
        - need to deal with outputs, make sure these get outputs
        - even though not fit
    - make guesses in arcsec (put guesser into this code don't use
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
from . import files

logger = logging.getLogger(__name__)

class Processor(object):
    """
    class to process a set of observations.
    """
    def __init__(self, args):
        self.args=args

        self._set_rng()
        self._load_conf()
        self._load_meds_files()
        self._load_fofs()
        self._set_fof_range()
        self._set_fitter()

    def go(self):
        """
        process the requested FoF groups
        """
        olist=[]
        elist=[]

        for fofid in range(self.start,self.end+1):
            logger.info('processing: %d:%d' % (fofid,self.end))
            output, epochs_data = self._process_fof(fofid)
            olist.append(output)
            if epochs_data is not None:
                elist.append(epochs_data)

        output = eu.numpy_util.combine_arrlist(olist)
        if len(elist) > 0:
            epochs_data = eu.numpy_util.combine_arrlist(elist)
        else:
            epochs_data = None


        self._write_output(output, epochs_data)

    def _process_fof(self, fofid):
        """
        process single FoF group
        """
        w,=np.where(self.fofs['fofid'] == fofid)
        logger.debug('%d objects' % w.size)
        assert w.size > 0,'no objects found for FoF id %d' % fofid

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

        output, epochs_data = self.fitter.go(mbobs_list)
        output['id'] = self.mb_meds.mlist[0]['id'][indices]
        output['fof_id'] = fofid
        return output, epochs_data


    def _write_output(self, output, epochs_data):
        """
        write the output as well as information from the epochs
        """
        logger.info('writing output: %s' % self.args.output)
        with fitsio.FITS(self.args.output,'rw',clobber=True) as fits:
            fits.write(output, extname='model_fits')
            if epochs_data is not None:
                fits.write(epochs_data, extname='epochs_data')

    def _set_rng(self):
        """
        set the rng given the input seed
        """
        self.rng = np.random.RandomState(self.args.seed)

    def _load_conf(self):
        """
        load the yaml config
        """
        logger.info('loading config: %s' % self.args.config)
        with open(self.args.config) as fobj:
            self.config = yaml.load(fobj)

    def _set_fitter(self):
        """
        currently only MOF
        """
        self.fitter = fitting.MOFFitter(
            self.config,
            self.mb_meds.nband,
            self.rng,
        ) 

    def _load_fofs(self):
        """
        load FoF group data from the input file
        """
        nbrs, fofs = files.load_fofs(self.args.fofs)
        self.fofs = fofs

    def _set_fof_range(self):
        """
        set the FoF range to be processed
        """
        nfofs = self.fofs['fofid'].max()+1
        assert nfofs == np.unique(self.fofs['fofid']).size

        self.start=self.args.start
        self.end=self.args.end

        if self.start is None:
            self.start = 0

        if self.end is None:
            self.end = nfofs-1

        print('processing fof range:',self.start,self.end)
        if self.start < 0 or self.end >= nfofs:
            mess='FoF range: [%d,%d] out of bounds [%d,%d]'
            mess = mess % (self.start,self.end,0,nfofs-1)
            raise ValueError(mess)

    def _load_meds_files(self):
        """
        load all MEDS files
        """
        mlist=[]
        for f in self.args.meds:
            logger.info('loading meds: %s' % f)
            mlist.append( ngmix.medsreaders.NGMixMEDS(f) )

        self.mb_meds = ngmix.medsreaders.MultiBandNGMixMEDS(mlist)
