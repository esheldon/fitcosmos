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

from .util import Namer

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
            assert fp['type'] == 'normal','only normal prior supported for fracdev'

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

            #for mbobs in mbobs_list:
            #    for bobs in mbobs:
            #        Tpsf=bobs[0].psf.gmix.get_T()
            #        scale=bobs[0].psf.jacobian.scale
            #        bobs.meta['T'] = Tpsf/scale**2

            mofc = self['mof']
            guess_from_priors=mofc.get('guess_from_priors',False)
            fitter = mof.MOFStamps(
                mbobs_list,
                mofc['model'],
                prior=self.mof_prior,
            )
            for i in range(ntry):
                guess=mof.moflib.get_stamp_guesses(
                    mbobs_list,
                    mofc['detband'],
                    mofc['model'],
                    self.rng,
                    prior=self.mof_prior,
                    guess_from_priors=guess_from_priors,
                )
                fitter.go(guess)

                res=fitter.get_result()
                if res['flags']==0:
                    break

        except BootPSFFailure as err:
            print(str(err))
            res={'flags':1}

        if res['flags'] != 0:
            fitter=None
            data=None
        else:
            average_fof_shapes = self.get('average_fof_shapes',False)
            if average_fof_shapes:
                logger.debug('averaging fof shapes')
                resavg=fitter.get_result_averaged_shapes()
                data=self._get_output(mbobs_list[0],[resavg], fitter.nband)
            else:
                reslist=fitter.get_result_list()
                data=self._get_output(mbobs_list[0], reslist, fitter.nband)

        if get_fitter:
            return fitter, data
        else:
            return data


    def _get_dtype(self, npars, nband):
        n=Namer(front=self['mof']['model'])
        dt = [
            ('fof_id','i4'), # fof id within image
            ('psf_g','f8',2),
            ('psf_T','f8'),
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
            (n('flux_cov'),'f8',(nband,nband)),
            (n('flux_err'),'f8',nband),
        ]

        if self['mof']['model']=='bdf':
            dt += [
                (n('fracdev'),'f8'),
                (n('fracdev_err'),'f8'),
            ]
        return dt

    def _get_output(self, mbobs_example, reslist,nband):

        npars=reslist[0]['pars'].size

        model=self['mof']['model']
        n=Namer(front=model)

        dt=self._get_dtype(npars, nband)
        output=np.zeros(len(reslist), dtype=dt)

        meta=mbobs_example.meta
        output['fof_id'] = meta['fof_id']

        for i,res in enumerate(reslist):
            t=output[i] 

            for name,val in res.items():
                if name=='nband':
                    continue

                if 'psf' in name:
                    t[name] = val
                else:
                    nname=n(name)
                    t[nname] = val

        return output

def _fit_all_psfs(mbobs_list, psf_conf):
    """
    fit all psfs in the input observations
    """
    fitter=AllPSFFitter(mbobs_list, psf_conf)
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


