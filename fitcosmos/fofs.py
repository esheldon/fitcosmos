import mof

def get_fofs(meds_list, fof_conf):
    mn=mof.fofs.MEDSNbrs(
        meds_list,
        fof_conf,
    )

    nbr_data = mn.get_nbrs()

    nf = mof.fofs.NbrsFoF(nbr_data)
    fofs = nf.get_fofs()

    return nbr_data, fofs
