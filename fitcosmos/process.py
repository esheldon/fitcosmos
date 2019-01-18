"""
TODO:

    - check mask frac on each stamp, reject if exceeds configurable maximum
    - note we are recording overall maskfrac

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
import time
from . import vis
from . import util

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

        tm0 = time.time()
        nfofs = self.end-self.start+1

        for fofid in range(self.start,self.end+1):
            logger.info('processing: %d:%d' % (fofid,self.end))

            tp = time.time()
            output, epochs_data = self._process_fof(fofid)
            tp = time.time()-tp
            logger.info('FoF time: %g' % tp)

            olist.append(output)
            if epochs_data is not None:
                elist.append(epochs_data)

        output = eu.numpy_util.combine_arrlist(olist)
        if len(elist) > 0:
            epochs_data = eu.numpy_util.combine_arrlist(elist)
        else:
            epochs_data = None

        tm = time.time()-tm0
        print('total time: %g' % tm)
        print('time per: %g' % (tm/nfofs))

        self._write_output(output, epochs_data)

    def _process_fof(self, fofid):
        """
        process single FoF group
        """
        w,=np.where(self.fofs['fofid'] == fofid)
        logger.info('FoF size: %d' % w.size)
        assert w.size > 0,'no objects found for FoF id %d' % fofid

        indices=self.fofs['number'][w]-1

        logger.debug('loading data')
        mbobs_list = self._get_fof_mbobs_list(indices)

        if self.args.save or self.args.show:
            self._doplots(fofid, mbobs_list)

        logger.debug('doing fits')
        output, epochs_data = self.fitter.go(mbobs_list)

        if self.args.save or self.args.show:
            self._doplots_compare_model(fofid, mbobs_list)

        self._add_extra_outputs(indices, output, fofid)
        return output, epochs_data

    def _add_extra_outputs(self, indices, output, fofid):

        m = self.mb_meds.mlist[0]
        output['id'] = m['id'][indices]
        output['ra'] = m['ra'][indices]
        output['dec'] = m['dec'][indices]
        output['flux_auto'] = m['flux_auto'][indices]
        output['mag_auto'] = m['mag_auto'][indices]
        output['fof_id'] = fofid

    def _get_fof_mbobs_list(self, indices):
        """
        load the mbobs_list for the input FoF group list
        """
        mbobs_list=[]
        for index in indices:
            mbobs = self._get_mbobs(index)
            mbobs_list.append(mbobs)

        return mbobs_list

    def _get_mbobs(self, index):
        mbobs=self.mb_meds.get_mbobs(
            index,
            weight_type='weight',
        )

        if self.config.get('inject',False):
            self._inject_fake_objects(mbobs)

        if self.config['keep_best_epoch']:
            mbobs = self._get_best_epochs(index, mbobs)

        self._set_weight(mbobs, index)

        mbobs.meta['masked_frac'] = util.get_masked_frac(mbobs)

        for band,obslist in enumerate(mbobs):
            m=self.mb_meds.mlist[band]
            meta = {
                'Tsky': 2* (m['iso_radius_arcsec'][index]*0.5)**2,
                #'Tsky': 0.1,
                'flux': m['flux_auto'][index],
                'magzp_ref': self.magzp_refs[band],
            }

            obslist.meta.update(meta)

            # fudge for ngmix working in surface brightness
            if self.config['parspace']=='ngmix':
                for obs in obslist:
                    pixel_scale2 = obs.jacobian.get_det()
                    pixel_scale4 = pixel_scale2*pixel_scale2
                    obs.image *= 1/pixel_scale2
                    obs.weight *= pixel_scale4

        return mbobs

    def _inject_fake_objects(self, mbobs):
        """
        inject a simple model for quick tests
        """
        import galsim
        # star
        #hlr=1.0e-5
        #flux=100.0
        hlr=0.1
        flux=10000.0

        #model = galsim.Exponential(
        #model = galsim.DeVaucouleurs(
        #    half_light_radius=hlr,
        #    flux=flux,
        #)
        model = galsim.Add(
            galsim.Exponential(
                half_light_radius=hlr,
                flux=0.5,
            ),
            galsim.DeVaucouleurs(
                half_light_radius=hlr,
                flux=0.5,
            )
        ).withFlux(flux)

        interp='lanczos15'

        for obslist in mbobs:
            for obs in obslist:
                psf_gsimage = galsim.Image(
                    obs.psf.image/obs.psf.image.sum(),
                    wcs=obs.psf.jacobian.get_galsim_wcs(),
                )

                psf_ii = galsim.InterpolatedImage(
                    psf_gsimage,
                    x_interpolant=interp,
                )

                gsimage = galsim.Image(
                    obs.image,
                    wcs=obs.jacobian.get_galsim_wcs(),
                )
                tmodel = galsim.Convolve(
                    model,
                    psf_ii,
                )
                tmodel.drawImage(
                    image=gsimage,
                    method='no_pixel',
                )

                image = gsimage.array

                wtmax = obs.weight.max()
                err = np.sqrt(1.0/wtmax)

                image += self.rng.normal(
                    scale=err,
                    size=image.shape,
                )

                obs.image = image


    def _get_best_epochs(self, index, mbobs):
        """
        just keep the best epoch if there are more than one

        this is good when using coadds and more than one epoch
        means overlap
        """
        new_mbobs=ngmix.MultiBandObsList()
        new_mbobs.meta.update(mbobs.meta)

        for band,obslist in enumerate(mbobs):
            nepoch=len(obslist)
            if nepoch > 1:

                mess='    obj %d band %d keeping best of %d epochs'
                logger.debug(mess % (index, band,nepoch))

                wts=np.array([ obs.weight.sum() for obs in obslist])
                logger.debug('    weights: %s' % str(wts))
                ibest=wts.argmax()
                keep_obs = obslist[ibest]

                new_obslist=ngmix.ObsList()
                new_obslist.meta.update(obslist.meta)
                new_obslist.append(keep_obs)
            else:
                new_obslist = obslist

            new_mbobs.append(new_obslist)
        return new_mbobs

    def _set_weight(self, mbobs, index):
        """
        set the weight

        we set a circular mask based on the radius.  For non hst bands
        we add quadratically with a fake psf fwhm of 1.5 arcsec
        """

        assert self.config['weight_type'] in ('weight','circular-mask')

        if self.config['weight_type'] == 'weight':
            return

        # hst_band can be None if we are only processing non-hst data
        hst_band=self.config['hst_band']

        fwhm=1.5
        sigma=fwhm/2.35
        exrad=3*sigma

        for band,obslist in enumerate(mbobs):
            m=self.mb_meds.mlist[band]
            rad = m['iso_radius_arcsec'][index]*3.0

            if band != hst_band:
                #rad = np.sqrt(rad**2 + 0.4**2)
                rad = np.sqrt(rad**2 + exrad**2)

            for obs in obslist:
                imshape=obs.image.shape
                jac = obs.jacobian
                scale = jac.scale
                rad_pix = rad/scale
                rad_pix2 = rad_pix**2

                rows, cols = np.mgrid[
                    0:imshape[0],
                    0:imshape[1],
                ]
                #cen = (np.array(imshape)-1.0)/2.0
                cen = jac.cen
                rows = rows.astype('f4') - cen[0]
                cols = cols.astype('f4') - cen[1]
                rad2 = rows**2 + cols**2
                w=np.where(rad2 > rad_pix2)
                if w[0].size > 0:
                    twt = obs.weight.copy()
                    twt[w] = 0.0
                    obs.weight = twt

    def _doplots(self, fofid, mbobs_list):
        plt=vis.view_mbobs_list(mbobs_list, show=self.args.show)#, weight=True)
        if self.args.save:
            pltname='images-%06d.png' % fofid
            plt.title='FoF id: %d' % fofid
            logger.info('writing: %s' % pltname)
            plt.write(pltname,dpi=300)

    def _doplots_compare_model(self, fofid, mbobs_list):
        mof_fitter=self.fitter.get_mof_fitter()
        res=mof_fitter.get_result()
        if res['flags']==0:
            vis.compare_models(mbobs_list, mof_fitter)

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
        parspace = self.config['parspace']
        if parspace=='ngmix':
            self.fitter = fitting.MOFFitter(
                self.config,
                self.mb_meds.nband,
                self.rng,
            ) 
        elif parspace=='galsim':
            self.fitter = fitting.MOFFitterGS(
                self.config,
                self.mb_meds.nband,
                self.rng,
            ) 
        else:
            raise ValueError('bad parspace "%s", should be '
                             '"ngmix" or "galsim"')

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

        logger.info('processing fof range: %d:%d' % (self.start,self.end))
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

        self.magzp_refs = []
        for m in self.mb_meds.mlist:
            meta=m.get_meta()
            self.magzp_refs.append(meta['magzp_ref'][0])

