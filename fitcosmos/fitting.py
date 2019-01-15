import logging
import numpy as np
from numpy import array
from pprint import pprint

import esutil as eu
import ngmix
from ngmix.gexceptions import GMixRangeError
from ngmix.observation import Observation
from ngmix.gexceptions import GMixMaxIterEM
from ngmix.gmix import GMixModel
from ngmix.gexceptions import BootPSFFailure, BootGalFailure

from .util import Namer, NoDataError
from . import procflags

import mof

logger = logging.getLogger(__name__)

class FitterBase(dict):
    """
    base class for fitting
    """
    def __init__(self, conf, nband, rng):

        self.nband=nband
        self.rng=rng
        self.update(conf)
        self._setup()

    def go(self, mbobs_list):
        """
        do measurements.  This is abstract
        """
        raise NotImplementedError("implement go()")

    def _get_prior(self, conf):
        """
        Set all the priors
        """
        import ngmix
        from ngmix.joint_prior import PriorSimpleSep, PriorBDFSep

        if 'priors' not in conf:
            return None

        ppars=conf['priors']
        if ppars.get('prior_from_mof',False):
            return None

        # g
        gp = ppars['g']
        assert gp['type']=="ba"
        g_prior = self._get_prior_generic(gp)

        T_prior = self._get_prior_generic(ppars['T'])
        flux_prior = self._get_prior_generic(ppars['flux'])

        # center
        cp=ppars['cen']
        assert cp['type'] == 'normal2d'
        cen_prior = self._get_prior_generic(cp)

        if conf['model']=='bdf':
            assert 'fracdev' in ppars,"set fracdev prior for bdf model"
            fp = ppars['fracdev']
            assert 'normal' in fp['type'],'only normal type priors supported for fracdev'

            fracdev_prior = self._get_prior_generic(fp)

            prior = PriorBDFSep(
                cen_prior,
                g_prior,
                T_prior,
                fracdev_prior,
                [flux_prior]*self.nband,
            )

        else:

            prior = PriorSimpleSep(
                cen_prior,
                g_prior,
                T_prior,
                [flux_prior]*self.nband,
            )

        return prior

    def _get_prior_generic(self, ppars):
        """
        get a prior object using the input specification
        """
        ptype=ppars['type']

        if ptype=="flat":
            prior=ngmix.priors.FlatPrior(*ppars['pars'], rng=self.rng)

        elif ptype == 'two-sided-erf':
            prior=ngmix.priors.TwoSidedErf(*ppars['pars'], rng=self.rng)

        elif ptype=='normal':
            prior = ngmix.priors.Normal(
                ppars['mean'],
                ppars['sigma'],
                rng=self.rng,
            )

        elif ptype=='truncated-normal':
            prior = ngmix.priors.TruncatedGaussian(
                mean=ppars['mean'],
                sigma=ppars['sigma'],
                minval=ppars['minval'],
                maxval=ppars['maxval'],
                rng=self.rng,
            )

        elif ptype=='log-normal':
            prior = ngmix.priors.LogNormal(
                ppars['mean'],
                ppars['sigma'],
                rng=self.rng,
            )


        elif ptype=='normal2d':
            prior=ngmix.priors.CenPrior(
                0.0,
                0.0,
                ppars['sigma'],
                ppars['sigma'],
                rng=self.rng,
            )

        elif ptype=='ba':
            prior = ngmix.priors.GPriorBA(ppars['sigma'], rng=self.rng)

        else:
            raise ValueError("bad prior type: '%s'" % ptype)

        return prior

class MOFFitter(FitterBase):
    """
    class for multi-object fitting
    """
    def __init__(self, *args, **kw):

        super(MOFFitter,self).__init__(*args, **kw)

        self.mof_prior = self._get_prior(self['mof'])

    def go(self, mbobs_list, ntry=2, get_fitter=False):
        """
        run the multi object fitter

        parameters
        ----------
        mbobs_list: list of MultiBandObsList
            One for each object.  If it is a simple
            MultiBandObsList it will be converted
            to a list

        returns
        -------
        data: ndarray
            Array with all output fields
        """
        if not isinstance(mbobs_list,list):
            mbobs_list=[mbobs_list]

        try:
            _fit_all_psfs(mbobs_list, self['mof']['psf'])
            _measure_all_psf_fluxes(mbobs_list)

            epochs_data = self._get_epochs_output(mbobs_list)

            mofc = self['mof']
            fitter = mof.MOFStamps(
                mbobs_list,
                mofc['model'],
                prior=self.mof_prior,
            )
            for i in range(ntry):
                guess=get_stamp_guesses(
                    mbobs_list,
                    mofc['detband'],
                    mofc['model'],
                    self.rng,
                    prior=self.mof_prior,
                )
                fitter.go(guess)

                res=fitter.get_result()
                if res['flags']==0:
                    break

            if res['flags'] != 0:
                res['main_flags'] = procflags.OBJ_FAILURE
                res['main_flagstr'] = procflags.get_flagname(res['main_flags'])
            else:
                res['main_flags'] = 0
                res['main_flagstr'] = procflags.get_flagname(0)

        except NoDataError as err:
            epochs_data=None
            print(str(err))
            res={
                'main_flags':procflags.NO_DATA,
                'main_flagstr':procflags.get_flagname(procflags.NO_DATA),
            }

        except BootPSFFailure as err:
            epochs_data=None
            print(str(err))
            res={
                'main_flags':procflags.PSF_FAILURE,
                'main_flagstr':procflags.get_flagname(procflags.PSF_FAILURE),
            }

        nobj = len(mbobs_list)

        if res['main_flags'] != 0:
            reslist=None
        else:
            reslist=fitter.get_result_list()

        data=self._get_output(
            mbobs_list,
            res,
            reslist,
        )

        return data, epochs_data

    def _setup(self):
        """
        set some useful values
        """
        self.npars = self.get_npars()
        self.npars_psf = self.get_npars_psf()

    @property
    def model(self):
        """
        model for fitting
        """
        return self['mof']['model']

    def get_npars(self):
        """
        number of pars we expect
        """
        return ngmix.gmix.get_model_npars(self.model) + self.nband-1

    def get_npars_psf(self):
        model=self['mof']['psf']['model']
        return 6*ngmix.gmix.get_model_ngauss(model)

    @property
    def namer(self):
        return Namer(front=self['mof']['model'])

    def _get_epochs_dtype(self):
        dt = [
            ('id','i8'),
            ('band','i2'),
            ('file_id','i4'),
            ('psf_pars','f8',self.npars_psf),
        ]
        return dt

    def _get_epochs_struct(self):
        dt=self._get_epochs_dtype()
        return np.zeros(1, dtype=dt)

    def _get_epochs_output(self, mbobs_list):
        elist=[]
        for mbobs in mbobs_list:
            for band, obslist in enumerate(mbobs):
                for obs in obslist:
                    meta=obs.meta
                    edata = self._get_epochs_struct()
                    edata['id'] = meta['id']
                    edata['band'] = band
                    edata['file_id'] = meta['file_id']
                    psf_gmix = obs.psf.gmix
                    edata['psf_pars'][0] = psf_gmix.get_full_pars()

                    elist.append(edata)

        edata = eu.numpy_util.combine_arrlist(elist)
        return edata

    def _get_dtype(self):
        npars = self.npars
        nband = self.nband

        n=self.namer
        dt = [
            ('id','i8'),
            ('ra','f8'),
            ('dec','f8'),
            ('flux_auto','f4'),
            ('mag_auto','f4'),
            ('fof_id','i8'), # fof id within image
            ('flags','i4'),
            ('flagstr','U11'),
            ('psf_g','f8',2),
            ('psf_T','f8'),
            ('psf_flux_flags','i4',nband),
            ('psf_flux','f8',nband),
            ('psf_mag','f8',nband),
            ('psf_flux_err','f8',nband),
            ('psf_flux_s2n','f8',nband),
            (n('flags'),'i4'),
            (n('nfev'),'i4'),
            (n('s2n'),'f8'),
            (n('pars'),'f8',npars),
            (n('pars_cov'),'f8',(npars,npars)),
            (n('g'),'f8',2),
            (n('g_cov'),'f8',(2,2)),
            (n('T'),'f8'),
            (n('T_err'),'f8'),
            (n('T_ratio'),'f8'),
            (n('flux'),'f8',nband),
            (n('mag'),'f8',nband),
            (n('flux_cov'),'f8',(nband,nband)),
            (n('flux_err'),'f8',nband),
        ]

        if self['mof']['model']=='bdf':
            dt += [
                (n('fracdev'),'f8'),
                (n('fracdev_err'),'f8'),
            ]
        return dt

    def _get_struct(self, nobj):
        dt = self._get_dtype()
        st = np.zeros(nobj, dtype=dt)
        st['flags'] = procflags.NO_ATTEMPT
        st['flagstr'] = procflags.get_flagname(procflags.NO_ATTEMPT)

        n=self.namer
        st[n('flags')] = st['flags']

        return st

    def _get_output(self, mbobs_list, main_res, reslist):

        nband=self.nband
        nobj = len(mbobs_list)
        output=self._get_struct(nobj)

        output['flags'] = main_res['main_flags']
        output['flagstr'] = main_res['main_flagstr']

        n=self.namer
        pn=Namer(front='psf')

        if 'flags' in main_res:
            output[n('flags')] = main_res['flags']

        # model flags will remain at NO_ATTEMPT
        if main_res['main_flags'] == 0:


            for i,res in enumerate(reslist):
                t=output[i] 
                mbobs = mbobs_list[i]

                for band,obslist in enumerate(mbobs):
                    meta = obslist.meta

                    if nband > 1:
                        t['psf_flux_flags'][band] = meta['psf_flux_flags']
                        for name in ('flux','flux_err','flux_s2n'):
                            t[pn(name)][band] = meta[pn(name)]

                        tflux = t[pn('flux')][band].clip(min=0.001)
                        t[pn('mag')][band] = meta['magzp_ref']-2.5*np.log10(tflux)


                    else:
                        t['psf_flux_flags'] = meta['psf_flux_flags']
                        for name in ('flux','flux_err','flux_s2n'):
                            t[pn(name)] = meta[pn(name)]

                        tflux = t[pn('flux')].clip(min=0.001)
                        t[pn('mag')] = meta['magzp_ref']-2.5*np.log10(tflux)



                for name,val in res.items():
                    if name=='nband':
                        continue

                    if 'psf' in name:
                        t[name] = val
                    else:
                        nname=n(name)
                        t[nname] = val

                for band,obslist in enumerate(mbobs):
                    meta = obslist.meta
                    if nband > 1:
                        tflux = t[n('flux')][band].clip(min=0.001)
                        t[n('mag')][band] = meta['magzp_ref']-2.5*np.log10(tflux)
                    else:
                        tflux = t[n('flux')].clip(min=0.001)
                        t[n('mag')] = meta['magzp_ref']-2.5*np.log10(tflux)

        return output

def _fit_all_psfs(mbobs_list, psf_conf):
    """
    fit all psfs in the input observations
    """
    fitter=AllPSFFitter(mbobs_list, psf_conf)
    fitter.go()

def _measure_all_psf_fluxes(mbobs_list):
    """
    fit all psfs in the input observations
    """
    fitter=AllPSFFluxFitter(mbobs_list)
    fitter.go()


class AllPSFFitter(object):
    def __init__(self, mbobs_list, psf_conf):
        self.mbobs_list=mbobs_list
        self.psf_conf=psf_conf

    def go(self):
        for mbobs in self.mbobs_list:
            for obslist in mbobs:
                for obs in obslist:
                    psf_obs = obs.get_psf()
                    _fit_one_psf(psf_obs, self.psf_conf)

def _fit_one_psf(obs, pconf):
    Tguess=4.0*obs.jacobian.get_scale()**2

    if 'coellip' in pconf['model']:
        ngauss=ngmix.bootstrap.get_coellip_ngauss(pconf['model'])
        runner=ngmix.bootstrap.PSFRunnerCoellip(
            obs,
            Tguess,
            ngauss,
            pconf['lm_pars'],
        )


    else:
        runner=ngmix.bootstrap.PSFRunner(
            obs,
            pconf['model'],
            Tguess,
            pconf['lm_pars'],
        )

    runner.go(ntry=pconf['ntry'])

    psf_fitter = runner.fitter
    res=psf_fitter.get_result()
    obs.update_meta_data({'fitter':psf_fitter})

    if res['flags']==0:
        gmix=psf_fitter.get_gmix()
        obs.set_gmix(gmix)
    else:
        raise BootPSFFailure("failed to fit psfs: %s" % str(res))

class AllPSFFluxFitter(object):
    def __init__(self, mbobs_list):
        self.mbobs_list=mbobs_list

    def go(self):
        for mbobs in self.mbobs_list:
            for band,obslist in enumerate(mbobs):

                if len(obslist) == 0:
                    raise NoDataError('no data in band %d' % band)

                meta=obslist.meta

                res = self._fit_psf_flux(band,obslist)
                meta['psf_flux_flags'] = res['flags']

                for n in ('psf_flux','psf_flux_err','psf_flux_s2n'):
                    meta[n] = res[n.replace('psf_','')]

    def _fit_psf_flux(self, band, obslist):
        fitter=ngmix.fitting.TemplateFluxFitter(
            obslist,
            do_psf=True,
        )
        fitter.go()

        res=fitter.get_result()

        if res['flags'] == 0 and res['flux_err'] > 0:
            res['flux_s2n'] = res['flux']/res['flux_err']
        else:
            res['flux_s2n'] = -9999.0
            raise BootPSFFailure("failed to fit psf fluxes for band %d: %s" % (band,str(res)))

        return res

def get_stamp_guesses(list_of_obs,
                      detband,
                      model,
                      rng,
                      prior=None):
    """
    get a guess based on metadata in the obs

    T guess is gotten from detband
    """

    nband=len(list_of_obs[0])

    if model=='bdf':
        npars_per=6+nband
    else:
        npars_per=5+nband

    nobj=len(list_of_obs)

    npars_tot = nobj*npars_per
    guess = np.zeros(npars_tot)

    pos_range = 0.005
    for i,mbo in enumerate(list_of_obs):
        detobslist = mbo[detband]
        detmeta=detobslist.meta

        obs=detobslist[0]

        T=detmeta['Tsky']

        beg=i*npars_per

        # always close guess for center
        guess[beg+0] = rng.uniform(low=-pos_range, high=pos_range)
        guess[beg+1] = rng.uniform(low=-pos_range, high=pos_range)

        # always arbitrary guess for shape
        guess[beg+2] = rng.uniform(low=-0.05, high=0.05)
        guess[beg+3] = rng.uniform(low=-0.05, high=0.05)

        guess[beg+4] = T*(1.0 + rng.uniform(low=-0.05, high=0.05))

        # arbitrary guess for fracdev
        if model=='bdf':
            guess[beg+5] = rng.uniform(low=0.4,high=0.6)
            flux_start=6
        else:
            flux_start=5

        for band, obslist in enumerate(mbo):
            obslist=mbo[band]
            scale = obslist[0].jacobian.scale
            band_meta=obslist.meta

            # note we take out scale**2 in DES images when
            # loading from MEDS so this isn't needed
            flux=band_meta['psf_flux']
            flux_guess=flux*(1.0 + rng.uniform(low=-0.05, high=0.05))

            guess[beg+flux_start+band] = flux_guess

    return guess


