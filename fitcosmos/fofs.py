import numpy as np
import mof

def get_fofs(meds_list, fof_conf):
    mn=MEDSNbrs(
        meds_list,
        fof_conf,
    )

    nbr_data = mn.get_nbrs()

    nf = mof.fofs.NbrsFoF(nbr_data)
    fofs = nf.get_fofs()

    return nbr_data, fofs

def get_fofs_old(meds_list, fof_conf):
    mn=mof.fofs.MEDSNbrs(
        meds_list,
        fof_conf,
    )

    nbr_data = mn.get_nbrs()

    nf = mof.fofs.NbrsFoF(nbr_data)
    fofs = nf.get_fofs()

    return nbr_data, fofs

class MEDSNbrs(object):
    """
    Gets nbrs of any postage stamp in the MEDS.

    A nbr is defined as any stamp which overlaps the stamp under consideration
    given a buffer or is in the seg map. See the code below.

    Options:
        buff_type - how to compute buffer length for stamp overlap
            'min': minimum of two stamps
            'max': max of two stamps
            'tot': sum of two stamps

        buff_frac - fraction by whch to multiply the buffer

        maxsize_to_replace - postage stamp size to replace with maxsize
        maxsize - size ot use instead of maxsize_to_replace to compute overlap

        check_seg - use object's seg map to get nbrs in addition to postage stamp overlap
    """

    def __init__(self,meds,conf,cat=None):
        self.meds = meds
        self.conf = conf

        self._init_bounds()

    def _init_bounds(self):
        if self.conf['method'] == 'radius':
            return self._init_bounds_by_radius()
        else:
            raise NotImplementedError('stamps not implemented for ra,dec version')
            return self._init_bounds_by_stamps()

    def _init_bounds_by_radius(self):

        radius_name=self.conf['radius_column']

        min_radius=self.conf.get('min_radius_arcsec',None)
        if min_radius is None:
            # arcsec
            min_radius=1.0

        max_radius=self.conf.get('max_radius_arcsec',None)
        if max_radius is None:
            max_radius=np.inf

        m=self.meds

        med_ra = np.median( m['ra'] )
        med_dec = np.median( m['dec'] )

        r = m[radius_name].copy()

        r *= self.conf['radius_mult']

        r.clip(min=min_radius, max=max_radius, out=r)

        r += self.conf['padding_arcsec']

        # factor of 2 because this should be a diameter as it is used later
        diameter = r*2
        self.sze = diameter

        ra_diff = (m['ra'] - med_ra)*3600.0
        dec_diff = (m['dec'] - med_dec)*3600.0

        self.l = ra_diff - r
        self.r = ra_diff + r
        self.b = dec_diff - r
        self.t = dec_diff + r


    def get_nbrs(self,verbose=True):
        #data types
        nbrs_data = []
        dtype = [('number','i8'),('nbr_number','i8')]
        #print("config:",self.conf)

        for mindex in range(self.meds.size):
            nbrs = []
            nbrs = self.check_mindex(mindex)

            nbrs = np.unique(nbrs)

            #add to final list
            for nbr in nbrs:
                nbrs_data.append((self.meds['number'][mindex],nbr))

        #return array sorted by number
        nbrs_data = np.array(nbrs_data,dtype=dtype)
        i = np.argsort(nbrs_data['number'])
        nbrs_data = nbrs_data[i]

        return nbrs_data

    def check_mindex(self,mindex):
        m = self.meds

        #check that current gal has OK stamp, or return bad crap
        if (m['orig_start_row'][mindex,0] == -9999
                or m['orig_start_col'][mindex,0] == -9999):

            nbr_numbers = np.array([-1],dtype=int)
            return nbr_numbers

        nbr_numbers = []

        #box intersection test and exclude yourself
        #use buffer of 1/4 of smaller of pair
        # sze is a diameter


        q, = np.where((~((self.l[mindex] > self.r) | (self.r[mindex] < self.l) |
                            (self.t[mindex] < self.b) | (self.b[mindex] > self.t))) &
                         (m['number'][mindex] != m['number']) &
                         (m['orig_start_row'][:,0] != -9999) & (m['orig_start_col'][:,0] != -9999))

        if len(q) > 0:
            nbr_numbers.extend(list(m['number'][q]))

        #cut weird crap
        if len(nbr_numbers) > 0:
            nbr_numbers = np.array(nbr_numbers,dtype=int)
            nbr_numbers = np.unique(nbr_numbers)
            inds = nbr_numbers-1
            q, = np.where((m['orig_start_row'][inds,0] != -9999) & (m['orig_start_col'][inds,0] != -9999))
            if len(q) > 0:
                nbr_numbers = list(nbr_numbers[q])
            else:
                nbr_numbers = []

        #if have stuff return unique else return -1
        if len(nbr_numbers) == 0:
            nbr_numbers = np.array([-1],dtype=int)
        else:
            nbr_numbers = np.array(nbr_numbers,dtype=int)
            nbr_numbers = np.unique(nbr_numbers)

        return nbr_numbers


